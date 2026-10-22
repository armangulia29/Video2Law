use rfd::FileDialog;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: rfd (File Dialog Builder) ===");

    // Construct the dialog with filters matching our app's requirements
    let dialog = FileDialog::new()
        .add_filter("Video files", &["mp4", "mov", "avi", "mkv"])
        .set_title("Video2Law - Select swinging clip")
        .set_directory("clips");

    println!("Constructed FileDialog with video filters successfully.");
    println!("PASS: rfd dialog");
    Ok(())
}
