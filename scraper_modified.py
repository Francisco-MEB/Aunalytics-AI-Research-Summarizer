import re
import requests # to connect to the web
from bs4 import BeautifulSoup # parse HTML
from urllib.parse import urlparse, parse_qs # parse URL
from scholarly import scholarly # google scholar
from playwright.sync.api import sync_playwright # playwright API (in case no scholar link  is found)
# documentation scholarly https://scholarly.readthedocs.io/en/stable/quickstart.html

def is_valid_name(name):
    """Check if extracted text looks like a real name"""
    if not name or len(name) > 50 or len(name) < 5:
        return False
    
    # Filter out common navigation/menu terms
    invalid_terms = ['menu', 'navigation', 'home', 'about', 'contact', 'site', 
                     'page', 'header', 'footer', 'main', 'content', 'skip']
    
    name_lower = name.lower()
    for term in invalid_terms:
        if term in name_lower:
            return False
    
    # Should have at least 2 words (first and last name)
    words = name.split()
    if len(words) < 2:
        return False
    
    # Check if it looks like a name (mostly letters)
    if not re.match(r'^[A-Za-z\s\-\.]+$', name):
        return False
    
    return True

def extract_professor_name(websiteUrl):
    """Extract professor name from their website"""
    response = requests.get(websiteUrl, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract from URL first (most reliable for sites like nd.edu/taeho-jung)
    url_parts = websiteUrl.split('/')
    for part in url_parts:
        # FIXED: Changed logic - should be NOT startswith('www')
        if (part and 
            not part.startswith('html') and 
            not part.startswith('www') and  # FIXED: Added 'not'
            'edu' not in part and
            '.' not in part and 
            len(part) > 2):
            potential_name = part.replace('~', '').replace('-', ' ').title()
            if potential_name and is_valid_name(potential_name):
                print(f"Extracted from URL: {potential_name}")
                return potential_name

    # Look for <h1> or <h2> tags
    for tag in ['h1', 'h2']:
        header = soup.find(tag)
        if header:
            name = header.get_text(strip=True)
            if is_valid_name(name):  # IMPROVED: Use validation function
                print(f"Found name in <{tag}>: {name}")
                return name
    
    # Look in meta tags
    meta_author = soup.find('meta', attrs={'name': 'author'})
    if meta_author and meta_author.get('content'):
        name = meta_author.get('content')
        if is_valid_name(name):  # IMPROVED: Use validation function
            print(f"Found name in meta tag: {name}")
            return name
    
    print("Could not extract professor name")
    return None

def search_scholar_by_name(professor_name, max_results=5):
    """Search Google Scholar for a professor by name"""
    print(f"Searching Google Scholar for: {professor_name}")
    
    search_query = scholarly.search_author(professor_name)
    results = []
    
    for i in range(max_results):
        author = next(search_query, None)
        if author is None:
            break
        results.append(author)
    
    if not results:
        print(f"No Google Scholar profile found for: {professor_name}")
        return None
    
    # Display results
    print(f"\nFound {len(results)} potential matches:")
    print("=" * 80)
    for i, author in enumerate(results, 1):
        print(f"\n{i}. {author['name']}")
        print(f"   Affiliation: {author.get('affiliation', 'N/A')}")
        print(f"   Email domain: {author.get('email_domain', 'N/A')}")
        print(f"   Interests: {', '.join(author.get('interests', [])[:3])}")
        print(f"   Scholar ID: {author['scholar_id']}")
    print("=" * 80)
    
    selected = results[0]
    print(f"\nUsing first result: {selected['name']}")
    return selected['scholar_id']


# pass the website url from the user
def getScholar(websiteUrl):
    response  = requests.get(websiteUrl, timeout=10) #only retrieve data, returns response object
    soup = BeautifulSoup(response.text, 'html.parser') #changes response objecct to sout 

    scholarUrl = None
    for link in soup.find_all('a'): # a tags in HTML
        href = link.get('href') # search all links
        if href and "scholar.google" in href.lower():#if links point to google scholar
            scholarUrl = href
            print(f"Found Scholar URL: {scholarUrl}")
            return href #first link found 
    print("No Google Scholar link found on page")
    return None


#  search Google Scholar for a professor by name and return their scholar ID.

# from the url scholar 
def extractAuthor(scholarUrl):
    parsedUrl = urlparse(scholarUrl) # returns tuple of ParseResult object
    # ex URL 
    # https://scholar.google.com/citations?user=j7wN3bYAAAAJ&hl=en
    #ParseResult(scheme='https', netloc='scholar.google.com', 
    #                     path='/citations', params='', 
    #                     query='user=ABC123&hl=en', fragment='')
    
    query_parameters = parse_qs(parsedUrl.query)
    if 'user' not in query_parameters:
            print("Error: No user ID found in URL")
            return None
    authorId = query_parameters['user'][0]  
    #access .query attribute from the object,
    # parsedUrl.query     'user=j7wN3bYAAAAJ&hl=en'
    #parse_qs converts to dict 
    # {'user': ['j7wN3bYAAAAJ'], 'hl': ['en']}, get the first element from user 
    print(f"Author ID: {authorId}")
    return authorId # 'j7wN3bYAAAAJ'


def abstractsGet(authorId):   
    # from authour id 'j7wN3bYAAAAJ', get a dic of author w 
    # {'affiliation': 'Professor of Vision Science, UC Berkeley',  
    # 'email_domain': '@berkeley.edu',  
    # 'filled': False,  
    # 'homepage': 'http://bankslab.berkeley.edu/',  
    # 'interests': ['vision science', 'psychology', 'human factors', 'neuroscience'], 
    # 'name': 'Martin Banks', 'organization': 11816294095661060495, 
    # 'scholar_id': 'Smr99uEAAAAJ', '
    # source': 'AUTHOR_PROFILE_PAGE'}
    author = scholarly.search_author_id(authorId)
    # testing
    # print("Author profile: ")
    
    # find publications
    scholarly.fill(author, sections=['publications'])

    descriptions = []
    num_papers = min(10, len(author['publications'])) #handles if more or less than 10 publications
    for i in range(num_papers): #max 10 publications
        pub = author['publications'][i] # dic of publications so
        print(f"Getting paper #{i+1} details")
        scholarly.fill(pub)

        if 'abstract' in pub['bib']:
            descriptions.append(pub['bib']['abstract'])
        else:
            descriptions.append("No descriptions available")

    return descriptions



if __name__ == "__main__":
    
# input from sites also does not work  

    #inputUrl = "https://www2.eecs.berkeley.edu/Faculty/Homepages/abbeel.html"
    inputUrl ="https://sites.nd.edu/taeho-jung"
    print("Searching Website:")

    foundUrl = getScholar(inputUrl)

    #POTENTIAL ISSUE IF NOT GOOGLE SCHOLAR LINK FOUND QWOULD BE COOL TO SEARCH NAME
    name = None
    if foundUrl is None:
        name = extract_professor_name(inputUrl)
        if name is  None:
            print("Cannot extract information") 
            exit()
        
    uId = None
    if (foundUrl):
        uId = extractAuthor(foundUrl)
    elif (name):
        uId = search_scholar_by_name(name)

    if uId is None:
        print("Could not obtain Scholar ID - exiting")
        exit()

    print("Fetching papers...")
    abstracts = abstractsGet(uId)
    if abstracts:
        for elements in abstracts:
            print(elements)