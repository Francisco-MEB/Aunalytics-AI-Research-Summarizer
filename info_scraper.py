import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import json


def fetch_page(url, timeout=10):
    """Fetch and parse the webpage"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        return soup
    except Exception as e:
        print(f"Error fetching page: {e}")
        return None


def extract_name_from_url(url):
    """Extract name from URL path like /taeho-jung/ -> Taeho Jung"""
    if not url:
        return ""
    path = urlparse(url).path
    segments = [s for s in path.split('/') if s]
    if segments:
        last_segment = segments[-1]
        
        # Remove file extensions (.html, .htm, .php, etc.)
        last_segment = re.sub(r'\.(html?|php|aspx?|jsp)$', '', last_segment, flags=re.I)
        
        # MUST have hyphens to be a name pattern (e.g., taeho-jung, bang-na-mi)
        # Single words like "mchaney1" are NOT name patterns
        if '-' not in last_segment:
            return ""
        
        # Check if it looks like a name (firstname-lastname or firstname-middle-lastname)
        parts = last_segment.split('-')
        if len(parts) >= 2:
            # Remove any parts that are just numbers
            parts = [p for p in parts if not p.isdigit()]
            
            if len(parts) >= 2:
                potential_name = ' '.join(p.capitalize() for p in parts if p)
                # Basic validation - 2-3 words, each starting with capital and purely alphabetic
                words = potential_name.split()
                if 2 <= len(words) <= 3 and all(w and w[0].isupper() and w.isalpha() for w in words):
                    return potential_name
    return ""

def extract_name(soup, url):
    """Extract professor name - prioritize URL, then page content"""
    
    def looks_like_name(text):
        """Check if text looks like a person's name"""
        if not text or len(text) < 5:
            return False
            
        words = text.split()
        
        # Should be 2-4 words
        if not (2 <= len(words) <= 4):
            return False
        
        # Each word should start with capital letter and be non-empty
        if not all(w and w[0].isupper() for w in words):
            return False
        
        # Reject all caps names (like "JOHN SMITH")
        if text.isupper():
            return False
        
        # Should not contain bad keywords
        bad_keywords = ['school', 'university', 'college', 'department', 'faculty', 
                       'home', 'about', 'education', 'welcome', 'research', 'page',
                       'professor', 'lab', 'group', 'center', 'institute']
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in bad_keywords):
            return False
        
        return True
    
    # PRIORITY 1: Extract from URL
    url_name = extract_name_from_url(url)
    if url_name and looks_like_name(url_name):
        return url_name
    
    # PRIORITY 2: Title tag
    title = soup.find('title')
    if title:
        title_text = title.get_text(strip=True)
        # Try splitting by common separators
        for sep in [' — ', ' - ', ' | ', ' : ', '|']:
            if sep in title_text:
                parts = title_text.split(sep)
                # Try first part
                candidate = parts[0].strip()
                if looks_like_name(candidate):
                    return candidate
                # Try second part if first fails
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    if looks_like_name(candidate):
                        return candidate
                break
    
    # PRIORITY 3: Meta author
    meta_author = soup.find('meta', {'name': 'author'})
    if meta_author and meta_author.get('content'):
        candidate = meta_author['content'].strip()
        if looks_like_name(candidate):
            return candidate
    
    # PRIORITY 4: H1 not in navigation
    for h1 in soup.find_all('h1', limit=10):
        if h1.find_parent(['nav', 'header', 'footer']):
            continue
        candidate = h1.get_text(strip=True)
        if looks_like_name(candidate):
            return candidate
    
    # PRIORITY 5: Look for specific name patterns in text
    page_text = soup.get_text()
    patterns = [
        r'I\s+am\s+(?:an?\s+)?(?:Associate\s+|Assistant\s+)?Professor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Professor\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
        r'Dr\.?\s+([A-Z][a-z]+(?:\s+[A-Z]\.?\s*)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
    ]
    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            candidate = match.group(1).strip()
            if looks_like_name(candidate):
                return candidate
    
    return ""


# BIO EXTRACTION — capture 2–4 real intro paragraphs
############################################################

def extract_bio(soup):
    container = soup.find('div', class_=re.compile(r'entry-content|content'))
    if container:
        paras = container.find_all('p', recursive=False)
        good = [p.get_text(strip=True) for p in paras if len(p.get_text(strip=True)) > 80]
        if len(good) >= 2:
            return " ".join(good[:4])

    # fallback
    longp = []
    for p in soup.find_all('p'):
        tx = p.get_text(strip=True)
        if len(tx) > 140:
            longp.append(tx)
    if longp:
        return " ".join(longp[:3])

    return ""


def extract_email(soup):
    """Extract email address"""
    # Check for mailto links
    mailto = soup.find('a', href=re.compile(r'^mailto:', re.I))
    if mailto:
        email = mailto['href'].replace('mailto:', '').strip()
        return email.split('?')[0]  # Remove query params
    
    # Search text for email pattern
    text = soup.get_text()
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        return emails[0]
    
    return ""


def extract_education(soup):
    """Extract education with proper formatting"""
    education = []
    
    # Look for education section header
    edu_header = soup.find(['h1', 'h2', 'h3', 'h4', 'strong'], 
                           string=re.compile(r'education|academic background|degrees', re.I))
    
    if edu_header:
        # Check for list
        ul = edu_header.find_next('ul')
        if ul:
            for li in ul.find_all('li', recursive=False):
                text = li.get_text(strip=True)
                # Must contain both degree and institution
                if (re.search(r'Ph\.?D|Doctor|Master|M\.?[SA]\.?|Bachelor|B\.?[SA]\.?', text, re.I) and
                    re.search(r'University|College|Institute', text, re.I) and
                    len(text) > 20):
                    education.append(text)
        
        # Check for text blocks with degree info
        if not education:
            for sibling in edu_header.find_next_siblings(['p', 'div'], limit=10):
                text = sibling.get_text(strip=True)
                # Split by newlines or periods for multiple degrees
                lines = [l.strip() for l in re.split(r'\n|(?<=\d{4})\.\s*', text) if l.strip()]
                for line in lines:
                    if (re.search(r'Ph\.?D|Doctor|Master|M\.?[SA]\.?|Bachelor|B\.?[SA]\.?', line, re.I) and
                        re.search(r'University|College|Institute', line, re.I) and
                        len(line) > 20):
                        education.append(line)
    
    # If not found in section, look in bio text
    if not education:
        page_text = soup.get_text()
        # Pattern: "I received my Ph.D. degree in X at Y in YEAR"
        pattern = r'((?:received|earned|obtained)\s+(?:my|her|his|a|the)\s+(?:Ph\.?D\.?|M\.?[SA]\.?|B\.?[SA]\.?)[^.]{20,150}(?:University|College|Institute)[^.]{0,50}(?:in\s+\d{4})?)'
        matches = re.findall(pattern, page_text, re.I)
        for match in matches[:3]:
            if len(match.strip()) > 25:
                education.append(match.strip())
    
    return education if education else []


def extract_research_interests(soup):
    """Extract research interests from text"""
    interests = []
    
    # Look for research section
    research_header = soup.find(['h1', 'h2', 'h3', 'h4'], 
                                string=re.compile(r'research interest|research area|research focus', re.I))
    
    if research_header:
        # Get following content
        for sibling in research_header.find_next_siblings(['p', 'ul', 'div'], limit=5):
            if sibling.name in ['h1', 'h2', 'h3', 'h4']:
                break
            
            if sibling.name == 'ul':
                for li in sibling.find_all('li', recursive=False):
                    text = li.get_text(strip=True)
                    if 10 < len(text) < 200:
                        interests.append(text)
            elif sibling.name == 'p':
                text = sibling.get_text(strip=True)
                if 20 < len(text) < 500:
                    interests.append(text)
    
    # Alternative: extract from bio mentioning research
    if not interests:
        bio = extract_bio(soup)
        if 'research' in bio.lower():
            # Look for sentence mentioning research focus
            sentences = re.split(r'[.!?]', bio)
            for sent in sentences:
                if re.search(r'research|focus|study|work', sent, re.I):
                    sent = sent.strip()
                    if 30 < len(sent) < 300:
                        interests.append(sent)
                        break
    
    return interests[:5]


def extract_office_hours(soup):
    """Extract office hours"""
    text = soup.get_text()
    match = re.search(r'office\s+hours?\s*:?\s*([^\n]{10,150})', text, re.I)
    if match:
        hours = match.group(1).strip()
        if re.search(r'\d{1,2}:\d{2}|am|pm|monday|tuesday|wednesday|thursday|friday|appointment', hours, re.I):
            return hours
    return ""


def extract_courses(soup, base_url):
    """Extract courses - follow teaching link and parse table/list"""
    courses = []
    
    # First, look for a link to teaching page
    teaching_link = soup.find('a', string=re.compile(r'teaching|courses', re.I))
    teaching_url = None
    
    if teaching_link and teaching_link.get('href'):
        teaching_url = urljoin(base_url, teaching_link['href'])
        print(f"  → Following teaching link: {teaching_url}")
        teaching_soup = fetch_page(teaching_url)
        if teaching_soup:
            soup = teaching_soup  # Use teaching page
    
    # Look for teaching/courses section
    header = soup.find(['h1', 'h2', 'h3', 'h4'], 
                       string=re.compile(r'teaching|courses?|classes', re.I))
    
    if header:
        # Strategy 1: Table format (common)
        table = header.find_next('table')
        if table:
            for row in table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all(['td', 'th'])
                if cells:
                    # First cell usually has course code
                    course_text = cells[0].get_text(strip=True)
                    
                    # Extract course code (e.g., CSE 40622)
                    code_match = re.search(r'([A-Z]{2,4}\s*\d{3,5}[A-Z]?)', course_text)
                    
                    # Get full row text for title
                    full_text = ' '.join(c.get_text(strip=True) for c in cells)
                    
                    link = row.find('a')
                    
                    courses.append({
                        'code': code_match.group(1) if code_match else '',
                        'title': full_text[:200],
                        'link': urljoin(base_url, link['href']) if link and link.get('href') else ''
                    })
        
        # Strategy 2: List format
        if not courses:
            ul = header.find_next('ul')
            if ul:
                for li in ul.find_all('li', recursive=False):
                    text = li.get_text(strip=True)
                    code_match = re.search(r'([A-Z]{2,4}\s*\d{3,5}[A-Z]?)', text)
                    link = li.find('a')
                    
                    courses.append({
                        'code': code_match.group(1) if code_match else '',
                        'title': text[:200],
                        'link': urljoin(base_url, link['href']) if link and link.get('href') else ''
                    })
    
    # Remove duplicates by code
    seen = set()
    unique = []
    for course in courses:
        if course['code'] and course['code'] not in seen:
            seen.add(course['code'])
            unique.append(course)
        elif not course['code']:
            unique.append(course)
    
    return unique


def extract_important_links(soup, base_url):
    """Extract CV, LinkedIn, and other academic links"""
    links = {}
    
    # CV/Resume link
    cv_patterns = [r'\bcv\b', r'\bresume\b', r'\bvitae\b', r'curriculum']
    for pattern in cv_patterns:
        cv_link = soup.find('a', string=re.compile(pattern, re.I))
        if not cv_link:
            cv_link = soup.find('a', href=re.compile(pattern, re.I))
        
        if cv_link and cv_link.get('href'):
            links['cv'] = urljoin(base_url, cv_link['href'])
            break
    
    # GitHub
    github = soup.find('a', href=re.compile(r'github\.com', re.I))
    if github:
        links['github'] = github['href']
    
    # LinkedIn
    linkedin = soup.find('a', href=re.compile(r'linkedin\.com', re.I))
    if linkedin:
        links['linkedin'] = linkedin['href']
    
    return links


def scrape_professor_website(url):
    """Main scraping function"""
    print(f"\n{'='*60}")
    print(f"Scraping: {url}")
    print(f"{'='*60}\n")
    
    soup = fetch_page(url)
    if not soup:
        return None
    
    data = {
        'url': url,
        'name': '',
        'bio': '',
        'email': '',
        'education': [],
        'research_interests': [],
        'office_hours': '',
        'courses': [],
        'links': {}
    }
    
    print("✓ Extracting name...")
    data['name'] = extract_name(soup, url)
    print(f"  Found: {data['name'] or 'Not found'}")
    
    print("✓ Extracting email...")
    data['email'] = extract_email(soup)
    print(f"  Found: {data['email'] or 'Not found'}")
    
    print("✓ Extracting biography...")
    data['bio'] = extract_bio(soup)
    print(f"  Found: {len(data['bio'])} characters")
    
    print("✓ Extracting education...")
    data['education'] = extract_education(soup)
    print(f"  Found: {len(data['education'])} entries")
    
    print("✓ Extracting research interests...")
    data['research_interests'] = extract_research_interests(soup)
    print(f"  Found: {len(data['research_interests'])} interests")
    
    print("✓ Extracting office hours...")
    data['office_hours'] = extract_office_hours(soup)
    print(f"  Found: {data['office_hours'][:50] if data['office_hours'] else 'Not found'}")
    
    print("✓ Extracting courses...")
    data['courses'] = extract_courses(soup, url)
    print(f"  Found: {len(data['courses'])} courses")
    
    print("✓ Extracting important links...")
    data['links'] = extract_important_links(soup, url)
    print(f"  Found: {', '.join(data['links'].keys()) or 'None'}")
    
    return data


def save_to_json(data, filename='professor_data.json'):
    """Save scraped data to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*60}")
    print(f"✓ Data saved to {filename}")
    print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    urls = [
        "https://sites.nd.edu/taeho-jung/",
        "https://eeb.yale.edu/people/faculty/thomas-near",
        "https://education.indianapolis.iu.edu/faculty-research/faculty-directory/bang-na-mi.html",
    ] 
    
    for url in urls:
        data = scrape_professor_website(url)
        
        if data:
            print("\n=== SCRAPING RESULTS ===\n")
            print(json.dumps(data, indent=2, ensure_ascii=False))
            
            # Save with URL-based filename
            domain = urlparse(url).netloc.replace('.', '_')
            filename = f"professor_{domain}.json"
            save_to_json(data, filename)
        else:
            print(f"Failed to scrape {url}\n")
        
        print("\n" + "="*60 + "\n")