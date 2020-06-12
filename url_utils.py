from urllib.parse import urlparse
import tldextract

def get_netloc(url):
    parse = urlparse(url)
    return parse.netloc

def get_toplevel_domains(url):
    subdomain, domain, suffix = tldextract.extract(url)
    return domain, suffix
