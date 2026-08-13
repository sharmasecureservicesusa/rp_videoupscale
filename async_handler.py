# 5. Start FFmpeg pipe writer (Encodes upscaled frames & copies audio stream directly)
        writer = subprocess.Popen([
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{out_w}x{out_h}",
            "-pix_fmt", "bgr24",
            "-r", fps,
            "-i", "-",
            "-i", str(input_video),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "faster",
            "-movflags", "+faststart",
            "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "copy",
            str(output_video)
        ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)