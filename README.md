# spotilens
Extract songs from spotify playlist with OCR

# Required external libraries
* Tesseract OCR engine
* OpenCV library

# How to use
1. Make sure OpenCV and Tesseract libraries are installed
2. Open Spotify Web or Spotify Desktop Client
3. Open playlist to scan
4. Check if current playlist has subimages that can't be correctly recognized by OCR engine. Examples are E sign and Music Video sign next to the artist name
5. Make screenshot of each and add in the 'cut' folder
6. Run spotilens.py
7. Find Spotify window
8. Select points as shown on picture
<img width="1280" alt="Spotify_D6RPvTQw4b_red" src="https://github.com/user-attachments/assets/31e63f00-1073-41ca-ab09-36c5fbe5e277" />
9. Wait until script finishes
10. Results will be printed in playlist_N.txt and playlist_json_N.txt
