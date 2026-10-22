//! Shows the discovered equations as real typeset maths.
//!
//! Pipeline: the engine's Typst maths string → Typst compiles it to a page → SVG →
//! resvg draws the SVG into pixels → egui texture. Each equation is drawn once and
//! cached. If anything fails, we fall back to readable Unicode text.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use eframe::egui::{self, Color32, ColorImage, RichText, TextureHandle, TextureOptions, Ui};
use typst::diag::{FileError, FileResult};
use typst::foundations::{Bytes, Datetime, Duration};
use typst::syntax::{FileId, Source};
use typst::text::{Font, FontBook};
use typst::utils::LazyHash;
use typst::{Library, LibraryExt, World};

/// Renders mathematical equations discovered by the engine.
#[derive(Default)]
pub struct EquationView;

impl EquationView {
    /// Displays the discovered law of motion. `law_plain` also carries "(g ≈ …, L = …)".
    pub fn show_law(ui: &mut Ui, law_typst: &str, law_plain: &str) {
        ui.group(|ui| {
            ui.label(RichText::new("Discovered Law of Motion").strong().size(16.0));
            ui.add_space(4.0);

            if law_typst.is_empty() {
                // Nothing found yet, or an engine message such as "(clip too short …)".
                ui.label(RichText::new(law_plain).italics().color(Color32::GRAY));
                return;
            }
            show_maths(ui, law_typst, "#64dcff", 20.0);
            // The "(g ≈ 9.78 m/s², L = 0.50 m)" part of the plain text.
            if let Some(extra) = law_plain.find('(').map(|i| &law_plain[i..]) {
                ui.label(RichText::new(extra).size(14.0).color(Color32::LIGHT_GRAY));
            }
        });
    }

    /// Displays the conserved quantity "E = …" and its leak "dE/dt = …".
    /// The engine's strings already start with "E =" and "(dif E)/(dif t) =".
    pub fn show_conservation(ui: &mut Ui, conserved_typst: Option<&str>, leak_typst: Option<&str>) {
        ui.group(|ui| {
            ui.label(RichText::new("Conserved Quantity & Energy Leak").strong().size(15.0));
            ui.add_space(4.0);

            match conserved_typst {
                Some(conserved) => show_maths(ui, conserved, "#78ff96", 17.0),
                None => {
                    ui.label(RichText::new("No conserved quantity computed yet").italics().color(Color32::GRAY));
                }
            }
            if let Some(leak) = leak_typst {
                ui.add_space(2.0);
                show_maths(ui, leak, "#ffb464", 15.0);
            }
        });
    }
}

/// Draw one Typst maths string (as a cached picture), or Unicode text if Typst fails.
fn show_maths(ui: &mut Ui, maths: &str, colour: &str, size: f32) {
    static CACHE: OnceLock<Mutex<HashMap<String, Option<TextureHandle>>>> = OnceLock::new();
    let key = format!("{colour} {size} {maths}");
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    let texture = cache
        .entry(key.clone())
        .or_insert_with(|| render(maths, colour, size).map(|img| ui.ctx().load_texture(key, img, TextureOptions::LINEAR)));

    match texture {
        Some(tex) => {
            // Rendered at 2× for sharpness, so show it at half its pixel size.
            ui.add(egui::Image::new(&*tex).fit_to_exact_size(tex.size_vec2() / 2.0));
        }
        None => {
            let [r, g, b] = hex_rgb(colour);
            ui.label(RichText::new(typst_to_unicode(maths)).size(size).color(Color32::from_rgb(r, g, b)));
        }
    }
}

/// Typst maths → RGBA picture at 2× scale. None if compiling or drawing fails.
fn render(maths: &str, colour: &str, size: f32) -> Option<ColorImage> {
    let page_source = format!(
        "#set page(width: auto, height: auto, margin: 3pt, fill: none)\n#set text(size: {size}pt, fill: rgb(\"{colour}\"))\n$ {maths} $"
    );
    let world = MathsWorld::new(page_source);
    let document = typst::compile(&world).output.ok()?;
    let svg = typst_svg::svg_merged(&document, &Default::default(), Default::default());

    let tree = resvg::usvg::Tree::from_str(&svg, &resvg::usvg::Options::default()).ok()?;
    let scale = 2.0;
    let (w, h) = ((tree.size().width() * scale).ceil() as u32, (tree.size().height() * scale).ceil() as u32);
    let mut pixmap = resvg::tiny_skia::Pixmap::new(w.max(1), h.max(1))?;
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    Some(ColorImage::from_rgba_premultiplied([w as usize, h as usize], pixmap.data()))
}

/// The smallest Typst "world": one source file, the bundled fonts, no files, no clock.
struct MathsWorld {
    source: Source,
}

/// Typst's standard library and the fonts bundled with typst-assets (including a
/// maths font), loaded once and shared by every equation.
fn library() -> &'static LazyHash<Library> {
    static LIBRARY: OnceLock<LazyHash<Library>> = OnceLock::new();
    LIBRARY.get_or_init(|| LazyHash::new(Library::default()))
}

fn fonts() -> &'static (LazyHash<FontBook>, Vec<Font>) {
    static FONTS: OnceLock<(LazyHash<FontBook>, Vec<Font>)> = OnceLock::new();
    FONTS.get_or_init(|| {
        let fonts: Vec<Font> = typst_assets::fonts().flat_map(|data| Font::iter(Bytes::new(data))).collect();
        (LazyHash::new(FontBook::from_fonts(&fonts)), fonts)
    })
}

impl MathsWorld {
    fn new(text: String) -> Self {
        Self { source: Source::detached(text) }
    }
}

impl World for MathsWorld {
    fn library(&self) -> &LazyHash<Library> {
        library()
    }
    fn book(&self) -> &LazyHash<FontBook> {
        &fonts().0
    }
    fn main(&self) -> FileId {
        self.source.id()
    }
    fn source(&self, id: FileId) -> FileResult<Source> {
        if id == self.source.id() { Ok(self.source.clone()) } else { Err(FileError::AccessDenied) }
    }
    fn file(&self, _id: FileId) -> FileResult<Bytes> {
        Err(FileError::AccessDenied) // equations never load files
    }
    fn font(&self, index: usize) -> Option<Font> {
        fonts().1.get(index).cloned()
    }
    fn today(&self, _offset: Option<Duration>) -> Option<Datetime> {
        None
    }
}

fn hex_rgb(hex: &str) -> [u8; 3] {
    let v = u32::from_str_radix(hex.trim_start_matches('#'), 16).unwrap_or(0xffffff);
    [(v >> 16) as u8, (v >> 8) as u8, v as u8]
}

/// Turn the engine's Typst maths into readable Unicode, e.g.
/// "dot.double(theta) = -19.59 sin theta" → "θ̈ = −19.59 sin θ".
pub fn typst_to_unicode(typst: &str) -> String {
    let mut s = typst.to_string();
    for (from, to) in [
        ("(dif E)/(dif t)", "dE/dt"),
        ("dot.double(theta)", "θ̈"),
        ("abs(dot(theta))", "|θ̇|"),
        ("dot(theta)", "θ̇"),
        ("theta", "θ"),
        ("1/2 ", "½"),
        ("^2", "²"),
        ("^3", "³"),
        (" - ", " − "),
        ("= -", "= −"),
    ] {
        s = s.replace(from, to);
    }
    s
}
