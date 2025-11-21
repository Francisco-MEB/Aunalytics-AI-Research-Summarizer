import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from scholarly import scholarly

def getScholar(websiteUrl):
  
  url = input("Enter webstie URL (professor website): ")

  response  = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')

  scholarUrl = None
  for link in soup.find_all('a'):
      href = link.get('href')
      if href and "scholar.google" in href.lower():
          scholarUrl = href
          print(f"Found Scholar URL: {scholarUrl}")
          return href

def extractAuthor(scholarUrl):
   
   parsedUrl = urlparse(scholarUrl)
   authorId = parse_qs(parsedUrl.query)['user'][0]
   print(f"Author ID: {authorId}")
   return authorId

def abstractsGet(authorId):
   
   author = scholarly.search_author_id(authorId)
   print("Author profile: ")

   scholarly.fill(author, sections=['publications'])

   descriptions = []

   for i in range(10):
      pub = author['publications'][i]
      print(f"Getting paper #{i+1} details")
      scholarly.fill(pub)

      if 'abstract' in pub['bib']:
         descriptions.append(pub['bib']['abstract'])
      else:
         descriptions.append("No descriptions available")
      return descriptions

print(descriptions)
pass

if __name__ == "__main__":
   
   inputUrl = input("Enter professor website: ")

   print("Searching Website:")
   foundUrl = getScholar(inputUrl)

   if foundUrl:
      print(f"Found Google Scholar: {foundUrl}")

      uId = extractAuthor(foundUrl)

      print("Fetching papers...")
      abstracts = abstractsGet(uId)
