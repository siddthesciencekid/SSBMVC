import youtube_dl

# Downloads a youtube video to /data given a video id
def download_youtube_video(v_id, file_name):
    options = {
        'format': 'bestvideo[ext=mp4][height=720]/best[ext=mp4][height=720]',
        'outtmpl': 'data/' + file_name
    }

    with youtube_dl.YoutubeDL(options) as downloader:
        downloader.download([v_id])
