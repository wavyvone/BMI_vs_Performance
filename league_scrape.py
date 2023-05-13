import requests
from bs4 import BeautifulSoup
import wget
import os

def image_grab(url_link):
	'''
	Takes in url which is a string but I have no asserts
	'''
	response = requests.get(
		url=url_link,
	)
	soup = BeautifulSoup(response.content, 'html.parser')

	#use this for file path later
	title = soup.find(id="firstHeading").string.strip()

	#sanity check
	gallery_name = soup.find_all("div", {"class": "gallerytext"})
	name_list = []
	for name in gallery_name:
		name_list.append(name.get_text().strip())

	#grab links
	gallery_links = soup.find("ul", {"class": "gallery mw-gallery-traditional"}).find_all('a')
	img_list = []
	for img in gallery_links:
		img_list.append(img['href'])

	#might have to change this if youre a windows user
	path_str = "./" + title
	isExist = os.path.exists(path_str)
	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(path_str)

	#iterate list and download images
	for i in img_list:
		wget.download(i, out = path_str)


def placements_grab(url_link):
	'''
	Grab the tournament results however how to organize
	Takes in a url_link that is a string.
	'''


#image_grab("https://lol.fandom.com/wiki/Berserker_(Kim_Min-cheol)")
image_grab("https://lol.fandom.com/wiki/EMENES")