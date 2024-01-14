import launch

if not launch.is_installed('imageio_ffmpeg'):
    launch.run_pip('install imageio_ffmpeg')
