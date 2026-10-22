use nokhwa::query;
use nokhwa::utils::ApiBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Smoke Test: nokhwa (Camera list + one frame) ===");

    let cameras = query(ApiBackend::Auto)?;
    println!("Discovered {} camera device(s):", cameras.len());
    for (i, cam) in cameras.iter().enumerate() {
        println!(" [{}] Name: {}, Index: {}", i, cam.human_name(), cam.index());
    }

    if let Some(first) = cameras.first() {
        use nokhwa::Camera;
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{RequestedFormat, RequestedFormatType};

        println!("Attempting capture on: {}", first.human_name());
        let mut camera = Camera::new(
            first.index().clone(),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
        )?;
        camera.open_stream()?;
        let frame = camera.frame()?;
        let decoded = frame.decode_image::<RgbFormat>()?;
        println!("Captured frame: {}x{} pixels, buffer len: {}", decoded.width(), decoded.height(), decoded.len());
        assert!(decoded.len() > 0, "Decoded frame buffer is empty");
    } else {
        println!("No cameras detected on this device. Query API executed cleanly.");
    }

    println!("PASS: nokhwa camera list + one frame");
    Ok(())
}
