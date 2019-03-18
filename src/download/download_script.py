from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload() 

arguments = {
    "keywords":"sloughi, "+
    "afghan hound, "+
    "azawakh, "+
    "irish wolfhound, "+
    "magyar agar, "+
    "whippet, "+
    "piccolo levriero italiano, "+
    "saluki, "+
    "chart polski, "+
    "ruskaya psovaya borzaya, "+
    "deerhound, "+
    "galgo espanol, "+
    "greyhound",
    "limit":100,"print_urls":True
}

paths = response.download(arguments) 

print(paths) 