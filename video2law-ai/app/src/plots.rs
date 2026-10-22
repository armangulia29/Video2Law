use egui_plot::{Bar, BarChart, Line, Plot, PlotPoints, VLine};
use engine::Trajectory;

/// Renders the x(t) trajectory plot using `egui_plot`.
///
/// Optionally overlays the predicted "ghost" trajectory in a distinct style, and a grey
/// line where the training window ends (the ghost starts there).
pub fn show_x_t_plot(
    ui: &mut eframe::egui::Ui,
    traj: &Trajectory,
    ghost: Option<&Trajectory>,
    train_until: f64,
) {
    ui.vertical(|ui| {
        ui.label(eframe::egui::RichText::new("x(t) Motion").strong());

        let plot = Plot::new("x_t_trajectory_plot")
            .view_aspect(2.2)
            .allow_zoom(true)
            .allow_drag(true);

        plot.show(ui, |plot_ui| {
            // Plot tracked trajectory
            if !traj.t.is_empty() && traj.t.len() == traj.x.len() {
                let points = PlotPoints::from_iter(
                    traj.t.iter().zip(&traj.x).map(|(&t, &x)| [t, x]),
                );
                let line = Line::new("Tracked x(t)", points)
                    .color(eframe::egui::Color32::from_rgb(0, 180, 255))
                    .width(2.0);
                plot_ui.line(line);
                plot_ui.vline(VLine::new("Training ends", train_until).color(eframe::egui::Color32::GRAY));
            }

            // Plot predicted ghost trajectory if available
            if let Some(g) = ghost {
                if !g.t.is_empty() && g.t.len() == g.x.len() {
                    let ghost_points = PlotPoints::from_iter(
                        g.t.iter().zip(&g.x).map(|(&t, &x)| [t, x]),
                    );
                    let ghost_line = Line::new("Ghost x(t)", ghost_points)
                        .color(eframe::egui::Color32::from_rgba_unmultiplied(255, 100, 100, 180))
                        .width(2.0);
                    plot_ui.line(ghost_line);
                }
            }
        });
    });
}

/// Renders the phase portrait: swing angle θ against angular speed θ̇.
/// A closed loop means energy is conserved; a spiral means it's leaking.
///
/// Uses the engine's smoothed angle (same data the law was fitted on), not raw
/// frame-to-frame differences, which are far too noisy.
pub fn show_phase_portrait(ui: &mut eframe::egui::Ui, traj: &Trajectory, train_until: f64) {
    let Some(s) = engine::signal::pendulum_signal(traj, train_until) else { return };

    ui.vertical(|ui| {
        ui.label(eframe::egui::RichText::new("Phase Portrait (θ vs θ̇)").strong());

        let plot = Plot::new("phase_portrait_plot")
            .view_aspect(1.5)
            .allow_zoom(true)
            .allow_drag(true);

        plot.show(ui, |plot_ui| {
            let points = PlotPoints::from_iter(s.angle.iter().zip(&s.velocity).map(|(&a, &v)| [a, v]));
            let line = Line::new("Trajectory", points)
                .color(eframe::egui::Color32::from_rgb(80, 220, 120))
                .width(1.5);
            plot_ui.line(line);
        });
    });
}

/// Neptune Mode: the Hankel singular values as bars. Each moving object makes 2 tall
/// bars, so 4 tall bars from one visible card means a hidden partner.
pub fn show_singular_values(ui: &mut eframe::egui::Ui, values: &[f64]) {
    let bars: Vec<Bar> = values
        .iter()
        .take(8)
        .enumerate()
        .map(|(i, &v)| {
            let strong = v > 0.1; // same cut-off the engine uses to count objects
            let colour = if strong { eframe::egui::Color32::from_rgb(255, 120, 220) } else { eframe::egui::Color32::GRAY };
            Bar::new(i as f64 + 1.0, v).fill(colour)
        })
        .collect();
    Plot::new("singular_values_plot")
        .height(110.0)
        .allow_zoom(false)
        .allow_drag(false)
        .show(ui, |plot_ui| plot_ui.bar_chart(BarChart::new("σ (share of largest)", bars)));
}
