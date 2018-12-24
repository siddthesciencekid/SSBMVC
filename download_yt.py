import youtube_dl

# Downloads and saves a youtube video as an mp4
# file in 720p to the data/ directory
def download_youtube_video(v_id, file_name):
    options = {
        'format': 'bestvideo[ext=mp4][height=720]/best[ext=mp4][height=720]',
        'outtmpl': 'data/' + file_name
    }

    with youtube_dl.YoutubeDL(options) as downloader:
        downloader.download([v_id])
