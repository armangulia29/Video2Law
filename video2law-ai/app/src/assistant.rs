//! The chat / voice panel for the built-in AI assistant (the AI itself is in the `ai` crate).
//!
//! This file only shows the conversation and passes messages along. When the AI wants
//! the app to do something, `poll` hands the request to `main.rs`, which does it.

use std::sync::mpsc::{Receiver, Sender, TryRecvError};

use ai::voice::{Mic, Voice};
use ai::{Action, ActionResult, FromAi, ToAi};
use eframe::egui::{self, Color32, RichText};

enum Who {
    User,
    Assistant,
    Note, // status lines: loading, errors, "heard: …"
}

pub struct Assistant {
    link: Option<(Sender<ToAi>, Receiver<FromAi>)>, // None until the user starts the AI
    ready: bool,
    thinking: bool,
    chat: Vec<(Who, String)>,
    input: String,
    mic: Option<Mic>,
    voice: Option<Voice>,
    speak_replies: bool,
}

impl Default for Assistant {
    fn default() -> Self {
        Self {
            link: None,
            ready: false,
            thinking: false,
            chat: vec![(Who::Note, "Start the assistant to control the app by chat or voice.".into())],
            input: String::new(),
            mic: None,
            voice: None,
            speak_replies: true,
        }
    }
}

impl Assistant {
    /// Collect messages from the AI. Returns the actions it wants the app to perform.
    pub fn poll(&mut self) -> Vec<(Action, Sender<ActionResult>)> {
        let mut actions = Vec::new();
        let Some((_, rx)) = &self.link else { return actions };
        loop {
            match rx.try_recv() {
                Ok(FromAi::Loading(text)) => self.chat.push((Who::Note, text)),
                Ok(FromAi::Ready) => {
                    self.ready = true;
                    self.chat.push((Who::Note, "Assistant ready. Try: \"Open the swing clip and find g.\"".into()));
                }
                Ok(FromAi::Heard(text)) => self.chat.push((Who::User, text)),
                Ok(FromAi::Thinking) => self.thinking = true,
                Ok(FromAi::Reply(text)) => {
                    self.thinking = false;
                    if self.speak_replies {
                        if let Some(voice) = &mut self.voice {
                            voice.say(&text);
                        }
                    }
                    self.chat.push((Who::Assistant, text));
                }
                Ok(FromAi::Do(action, reply)) => actions.push((action, reply)),
                Ok(FromAi::Error(text)) => {
                    self.thinking = false;
                    self.chat.push((Who::Note, format!("⚠ {text}")));
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.chat.push((Who::Note, "The assistant stopped.".into()));
                    self.link = None;
                    self.ready = false;
                    break;
                }
            }
        }
        actions
    }

    /// True while the AI is loading or thinking (keep repainting so replies show up).
    pub fn is_active(&self) -> bool {
        self.link.is_some() && (!self.ready || self.thinking)
    }

    /// Are replies read aloud? (Part of the state the AI sees.)
    pub fn speaks(&self) -> bool {
        self.speak_replies
    }

    /// The AI can switch reading aloud on or off.
    pub fn set_speaks(&mut self, on: bool) {
        self.speak_replies = on;
        if !on {
            if let Some(voice) = &mut self.voice {
                voice.stop();
            }
        }
    }

    /// Note an action in the chat so the user sees what the AI is doing.
    pub fn note(&mut self, text: String) {
        self.chat.push((Who::Note, text));
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        ui.heading(RichText::new("🤖 Assistant").strong());
        if self.link.is_none() {
            if ui.button("▶ Start assistant").on_hover_text("Loads the local AI model (Qwen3-VL 4B). Takes a minute.").clicked() {
                self.link = Some(ai::start());
                self.voice = Voice::new().ok();
            }
        }
        ui.checkbox(&mut self.speak_replies, "Speak replies");
        ui.separator();

        // Conversation
        let input_height = 70.0;
        egui::ScrollArea::vertical()
            .max_height(ui.available_height() - input_height)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for (who, text) in &self.chat {
                    let (prefix, colour) = match who {
                        Who::User => ("You: ", Color32::from_rgb(140, 200, 255)),
                        Who::Assistant => ("AI: ", Color32::from_rgb(170, 255, 170)),
                        Who::Note => ("", Color32::GRAY),
                    };
                    ui.label(RichText::new(format!("{prefix}{text}")).color(colour));
                    ui.add_space(4.0);
                }
                if self.is_active() {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(if self.ready { "thinking…" } else { "loading…" });
                    });
                }
            });

        // Input: type, or hold the button and talk
        ui.separator();
        let can_send = self.ready && !self.thinking;
        ui.horizontal(|ui| {
            let edit = ui.add_enabled(can_send, egui::TextEdit::singleline(&mut self.input).hint_text("Ask or tell me what to do"));
            let enter = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if (ui.add_enabled(can_send, egui::Button::new("Send")).clicked() || enter) && !self.input.trim().is_empty() {
                let text = std::mem::take(&mut self.input);
                self.send(ToAi::Said(text.clone()));
                self.chat.push((Who::User, text));
            }
        });
        let talk = ui.add_enabled(can_send, egui::Button::new("🎤 Hold to talk").sense(egui::Sense::drag()));
        if talk.drag_started() {
            if let Some(voice) = &mut self.voice {
                voice.stop(); // don't record the assistant's own voice
            }
            match Mic::start() {
                Ok(mic) => self.mic = Some(mic),
                Err(e) => self.chat.push((Who::Note, format!("⚠ Microphone: {e}"))),
            }
        }
        if talk.drag_stopped() {
            if let Some(mic) = self.mic.take() {
                let samples = mic.stop();
                if samples.len() > 16_000 / 4 {
                    self.send(ToAi::Spoke(samples)); // at least a quarter of a second
                }
            }
        }
        if self.mic.is_some() {
            ui.label(RichText::new("● recording…").color(Color32::from_rgb(255, 90, 90)));
        }
    }

    fn send(&mut self, message: ToAi) {
        if let Some((tx, _)) = &self.link {
            let _ = tx.send(message);
        }
    }
}
