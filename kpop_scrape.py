import requests
from bs4 import BeautifulSoup
import wget
import os
import csv
import pandas as pd
from urllib.parse import urlparse, parse_qs
import re
 
def calculate_bmi(weight, height):
    #height and weight are can be strings

    try:
        bmi = (float(weight) / float(height) / float(height)) * 10000
        #print("BMI: ", bmi)
    except (ValueError, TypeError):
        bmi = 999   #Error
        #print("Error: Can't calculate BMI")
    return bmi

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
    
    img_info = info.find_all('img')
    img_info.pop(0) #pop first image because that's always to group image

    p_strip = [p.text.strip() for p in p_info]
    
    idols = {} #initialize dictionary
    names = [] #initalize an array to hold names
    for p in p_strip:
        if "Height" in p:
            arr = p.split('\n')
            for i in range(len(arr)): #Assume all idols have stage name, real name, height, and weight tabs and these tabs are in order
                if "Stage Name" in arr[i]:
                    s_name = arr[i].split(":")[1].split("(")[0].strip()
                    name = arr[i+1].split(":")[1].split("(")[0].strip()
                    names.append(f"{s_name} ({name})")
                    #print(f"{s_name} ({name})")
                if "Height" in arr[i]: #assume height (assume in cm) always followed by weight (assume in kg)
                    height = arr[i].split(":")[1].split("cm")[0].strip()
                    #print(height)
                    weight = arr[i+1].split(":")[1].split("kg")[0].strip()
                    #print(weight)
                    
                    #Calculate BMI and save into dictionary
                    idols[f"{s_name} ({name})"] = {"bmi": calculate_bmi(weight, height)}
                    #print("_______________________________________")  


    index = 0
    for image_tag in img_info: #assumes each idol/key in dictionary has image
        if idols[names[index]]["bmi"] == 999: #error, no BMI
            del idols[names[index]] #remove person from dictionary since no BMI
            index += 1 #update index
            continue
        
        # Get the image URL from the 'src' attribute
        image_url = image_tag['src']

        # Send a GET request to the image URL
        image_response = requests.get(image_url)

        # Check if the request was successful
        if image_response.status_code == 200:
            # Get the filename from the image URL
            #filename = os.path.basename(image_url)
            file_ext = ".jpg"
            filename = names[index].replace(" ", "_") + file_ext  

            # Specify the path to save the image
            save_directory = folder  # Update this with your desired directory name
            save_path = os.path.join(save_directory, filename) 
            idols[names[index]]["filename"] = filename #add filename corresponding to idol

            # Save the image to the specified path
            with open(save_path, 'wb') as file:
                file.write(image_response.content)
                #print(f"Image '{filename}' saved successfully.")
        else:
            print(f"Failed to download the image from URL: {image_url}")
        
        index += 1 #update index


    #Write dictionary into csv
    # Specify the field names for the CSV
    fieldnames = ["Name", "BMI", "Filename"]

    # Specify the name of the CSV file
    csv_filename = "idol_data.csv"

    # Open the CSV file in write mode
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data rows
        for person, data in idols.items():
            row = {"Name": person, "BMI": data["bmi"], "Filename": data["filename"]}
            writer.writerow(row)



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


if __name__ == "__main__":
    disbanded_groups = grab_all_url("https://kprofiles.com/disbanded-kpop-boy-groups/")
    active_groups = grab_all_url("https://kprofiles.com/k-pop-boy-groups/")
    total = active_groups + disbanded_groups
    for url in total:
        data_image_grab(url, "kpopimages")
    #data_image_grab("https://kprofiles.com/2nb-profile-facts/", "kpopimages")