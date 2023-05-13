import requests
from bs4 import BeautifulSoup
import wget
import os
import csv

def image_grab(url_link):
	'''
	Takes in url which is a string but I have no asserts
	'''
	response = requests.get(
		url=url_link,
	)
	if response.status_code != 200:
		print("BAD STATUS")
		return
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
	#DOES NOT CHECK DUPLICATES
	for i in img_list:
		wget.download(i, out = path_str)


def placements_grab(url_link):
	'''
	Grab the tournament results however how to organize
	Takes in a url_link that is a string.
	'''
	response = requests.get(
		url=url_link,
	)
	if response.status_code != 200:
		print("BAD STATUS")
		return

	soup = BeautifulSoup(response.content, 'html.parser')

	#use this for file path later
	title = soup.find(id="firstHeading").string.strip()
	dir_name = title[:title.index('/')]


	placements = soup.find("table", {"class": "wikitable sortable hoverable-rows"}).find_all('tr')
	placement_list = []
	for place in placements:
		place_l = place.get_text(",", strip=True).replace(',,,', ',').split(',')
		if len(place_l) > 15:
			#print(place_l)
			placement_list.append([place_l[1],place_l[2], place_l[9]])

	path_str = "./" + dir_name
	isExist = os.path.exists(path_str)
	if not isExist:
		# Create a new directory because it does not exist
		os.makedirs(path_str)

	file_str = title + ".csv"
	with open(file_str, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(placement_list)







#image_grab("https://lol.fandom.com/wiki/Berserker_(Kim_Min-cheol)")
#image_grab("https://lol.fandom.com/wiki/EMENES")
#placements_grab("https://lol.fandom.com/wiki/EMENES/Tournament_Results")