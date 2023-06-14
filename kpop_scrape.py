import requests
from bs4 import BeautifulSoup
import wget
import os
import csv
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re


def calculate_bmi(weight, height):
    # height and weight are can be strings

    try:
        bmi = (float(weight) / float(height) / float(height)) * 10000
        # print("BMI: ", bmi)
    except (ValueError, TypeError):
        bmi = 999  # Error
        # print("Error: Can't calculate BMI")
    return bmi


def grab_all_url(url_link):
    '''
    Grab all the urls from the main page of kprofiles
    just put in url links
    https://kprofiles.com/k-pop-boy-groups/
    https://kprofiles.com/disbanded-kpop-boy-groups/
    grabs all url links of the boy kpop profiles.
    '''
    try:
        response = requests.get(
            url=url_link,
        )
    except requests.exceptions.ConnectionError:
        print(f"Connection Error for {url_link} in image_grab")
        return url_link
    if response.status_code != 200:
        print("BAD STATUS image")
        return url_link
    soup = BeautifulSoup(response.content, 'html.parser')

    head = soup.find("div", class_="entry-content herald-entry-content")
    url_list = []
    for url in head.find_all("a", href=True)[1:]:
        url_list.append(url['href'])
    return url_list


def data_image_grab(url_link, folder):
    '''
    grabs all the photos out of a kpop page for
    creates a subfolder in the resident folder
    downloads all the pictures to the subfolder

    Params
    url_link- (str) in the form of https://kprofiles.com/ateez-members-profile/
    resident_folder- (str) in the form of  'Kpop/'
    '''

    try:
        response = requests.get(
            url=url_link,
        )
    except requests.exceptions.ConnectionError:
        print(f"Connection Error for {url_link} in image_grab")
        return url_link
    if response.status_code != 200:
        print("BAD STATUS image")
        return url_link

    soup = BeautifulSoup(response.content, 'html.parser')

    path_str = "./" + folder
    if not os.path.exists(path_str):
        # Create a new directory because it does not exist
        os.makedirs(path_str)

    info = soup.find("div", class_="entry-content herald-entry-content")
    p_info = info.find_all('p')

    img_info = []
    #img_info = info.find_all('img')
    #if len(img_info) > 1:  # more than one image
        #img_info.pop(0)  # pop first image
    # img_info.pop(0) #pop first image because that's always to group image

    # p_strip = [p.text.strip() for p in p_info]

    idols = {}  # initialize dictionary
    names = []  # initalize an array to hold namese
    members_prof = ""
    members_flag = False
    for strip_p in p_info:
        p = strip_p.text.strip()
        if "Height" in p:
            # since we know height is in here there has to be an image
            img_url = strip_p.find_all('img')
            if len(img_url) == 0:
                # if img_returns None then img doesnt exist
                continue
            else:
                for url in img_url:
                    if ".svg" not in url["src"]:
                        img_info.append(url["src"])
            arr = p.lower().split('\n')
            #print(arr)
            name_flag = False #if we have a name already
            for i in range(len(arr)):
                if not members_flag:
                    if "members:" in arr[i] or "member:" in arr[i] or "profile:" in arr[i]:
                        splitted = arr[i].split(" ")
                        for j in range(len(splitted)):
                            if "members" in splitted[j]:
                                members_prof = splitted[j-1]
                                members_flag = True
                                break

                if "name:" in arr[i] and not name_flag:
                    name_flag = True
                    s_name = arr[i].split(":")[1].split("(")[0].strip().replace('/', '_')
                    name = arr[i + 1].split(":")[1].split("(")[0].strip().replace('/', '_')
                    '''
                    if "name:" in arr[i + 1]:
                        name = arr[i + 1].split(":")[1].split("(")[0].strip().replace('/', '_')
                    else:
                        name = arr[i + 1].split(" ")[2].split("(")[0].strip().replace('/', '_')
                    '''
                    #print(f"{s_name} ({name})")
                    if s_name == "–":
                        s_name = name
                    if name == "–":
                        name = s_name
                    names.append(f"{members_prof} {s_name} ({name})")
                    #print(f"{s_name} ({name})")


                if "height:" in arr[i] or "height :" in arr[i]:  # assume height (assume in cm) always followed by weight (assume in kg)
                    height = arr[i].split(":")[1].split("cm")[0].strip()
                    weight = ""
                    if "weight:" in arr[i + 1]:
                        weight = arr[i + 1].split(":")[1].split("kg")[0].strip()
                    elif "weight" in arr[i + 1]:
                        weight = arr[i + 1].split(" ")[1].split("kg")[0].strip()
                    #print(weight)
                    #print(height , weight)
                    # Calculate BMI and save into dictionary
                    idols[f"{members_prof} {s_name} ({name})"] = {"bmi": calculate_bmi(weight, height)}

                    # print("_______________________________________")

    index = 0
    print(idols)
    #print(img_info)

    for image_tag in img_info:  # assumes each idol/key in dictionary has image
        if idols[names[index]]["bmi"] == 999:  # error, no BMI
            del idols[names[index]]  # remove person from dictionary since no BMI
            index += 1  # update index
            continue
        # Get the image URL from the 'src' attribute
        image_url = image_tag

        # Send a GET request to the image URL
        #print(image_url)
        image_response = requests.get(image_url)

        # Check if the request was successful
        if image_response.status_code == 200:
            # Get the filename from the image URL
            # filename = os.path.basename(image_url)
            file_ext = ".jpg"
            filename = names[index].replace(" ", "_") + file_ext

            # Specify the path to save the image
            save_directory = folder  # Update this with your desired directory name
            save_path = os.path.join(save_directory, filename)
            idols[names[index]]["filename"] = filename  # add filename corresponding to idol

            # Save the image to the specified path
            with open(save_path, 'wb') as file:
                file.write(image_response.content)
                # print(f"Image '{filename}' saved successfully.")
        else:
            print(f"Failed to download the image from URL: {image_url}")
            del idols[names[index]]

        index += 1  # update index


    #for key, value in idols.items():
        #print(f"Key: {key}, Value: {value}")


    # Write dictionary into csv
    # Specify the field names for the CSV
    fieldnames = ["Name", "BMI", "Filename"]

    # Specify the name of the CSV file
    csv_filename = "idol_data.csv"

    # Check if the file exists
    file_exists = os.path.isfile(csv_filename)

    # Open the CSV file in write mode
    with open(csv_filename, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row only if the file doesn't exist
        if not file_exists:
            writer.writeheader()
            
        # Write the data rows
        for person, data in idols.items():
            row = {"Name": person, "BMI": data["bmi"], "Filename": data["filename"]}
            writer.writerow(row)


if __name__ == "__main__":

    black_listed = ['https://kprofiles.com/rekids-dream-members-profile/', 'https://kprofiles.com/1-n-members-profile/', 'https://kprofiles.com/deep-studio-entertainment-boy-group-profile/',
                    'https://kprofiles.com/zerobaseone-members-profile/', 'https://kprofiles.com/2shai-members-profile/', 'https://kprofiles.com/ledapple-members-profile/']
    disbanded_groups = grab_all_url("https://kprofiles.com/disbanded-kpop-boy-groups/")
    active_groups = grab_all_url("https://kprofiles.com/k-pop-boy-groups/")
    total = active_groups + disbanded_groups
    for url in total:
        if url in black_listed:
            continue
        print("Scraping: ", url)
        data_image_grab(url, "kpopimages")
    '''
    data_image_grab("https://kprofiles.com/deep-studio-entertainment-boy-group-profile/", "kpopimages")
    '''