import requests
from bs4 import BeautifulSoup
import wget
import os
import csv

def image_grab(url_link, resident_folder):
	'''
	Takes in url which is a string but I have no asserts
	and resident_folder which is the dir it has
	e.g KR/ or NA/
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
	path_str = "./" + resident_folder + title
	isExist = os.path.exists(path_str)
	if isExist:
		return
	else:
		# Create a new directory because it does not exist
		os.makedirs(path_str)

	#iterate list and download images
	#DOES NOT CHECK DUPLICATES
	for i in img_list:
		wget.download(i, out = path_str)



def placements_grab(url_link, resident_folder):
	'''
	Grab the tournament results however how to organize
	Takes in a url_link that is a string.
	and resident_folder
	e.g. KR/
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
	dir_name = title.replace("/", "_").replace(" ","_")


	placements = soup.find("table", {"class": "wikitable sortable hoverable-rows"}).find_all('tr')
	placement_list = []
	for place in placements:
		place_l = place.get_text(",", strip=True).replace(',,,', ',').split(',')
		if len(place_l) > 15:
			#print(place_l)
			placement_list.append([place_l[1],place_l[2], place_l[9]])

	path_str = "./" + resident_folder + dir_name
	isExist = os.path.exists(path_str)
	if isExist:
		return
	else:
		# Create a new directory because it does not exist
		os.makedirs(path_str)

	file_str = path_str + "/"+ dir_name + ".csv"
	print(file_str)

	if os.path.isfile(file_str):
		return

	with open(file_str, "w", newline="") as f:
		writer = csv.writer(f)
		writer.writerows(placement_list)


def grab_all_players(url_link):
	'''
	takes in url_link of each region and collects all url of each player to feed in
	Should output
	Returns 2 lists, first one is list of names and the second is list of urls
	'''
	#append since the stuff from the table is limited
	wiki_url = "https://lol.fandom.com"
	response = requests.get(
		url=url_link,
	)
	if response.status_code != 200:
		print("BAD STATUS")
		return
	soup = BeautifulSoup(response.content, 'html.parser')

	url_list = []
	name_list = []

	url_table = soup.find("table", {"class": "cargoTable sortable"}).find_all('td', {"class": "field_ID"})
	for urls in url_table:
		name_list.append(urls.find('a').get_text())
		url_list.append(wiki_url + urls.find('a')["href"])

	assert len(url_list) == len(name_list)
	print(name_list)
	print(url_list)
	return name_list, url_list


def grab_all_residency(url_link):
	'''
	grab all residency links EXCEPT FOR EMA CUZ HUH
	Grab from the NA link
	'''
	wiki_url = "https://lol.fandom.com"
	response = requests.get(
		url=url_link,
	)
	if response.status_code != 200:
		print("BAD STATUS")
		return
	soup = BeautifulSoup(response.content, 'html.parser')
	#how to organize this?
	#get all the resident links first and store in list

	res_url_link = []
	resident_name = []

	residents = soup.find("div", {"class": "hlist"}).find_all('li')
	for resident in residents[1:]:
		res_url_link.append(wiki_url + resident.find('a')["href"])
		resident_name.append(resident.find('a').get_text() + "/")

	assert len(res_url_link) == len(resident_name)

	return resident_name, res_url_link


def grab_free_retired(url_link):



#image_grab("https://lol.fandom.com/wiki/Berserker_(Kim_Min-cheol)")
#image_grab("https://lol.fandom.com/wiki/EMENES", "KR/")
#placements_grab("https://lol.fandom.com/wiki/EMENES/Tournament_Results", "KR/")

#grab_all_players("https://lol.fandom.com/wiki/North_American_Players")
#grab_all_players("https://lol.fandom.com/wiki/North_American_Players/Free_Agents")

grab_all_residency("https://lol.fandom.com/wiki/North_American_Players")