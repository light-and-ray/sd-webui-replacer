import re, os
from dataclasses import dataclass
from replacer.options import EXT_ROOT_DIRECTORY
from modules import shared

JS_PREFIX = 'markdown_browser_javascript_'


@dataclass
class Anchor:
    name: str
    id: str
    depth: int


def getURLsFromFile(file: str) -> list[str]:
    urls = set()

    MDLinks = re.findall(r'\[.*?\]\((.+?)\)', file)
    for link in MDLinks:
        urls.add(link)

    srcLinks = re.findall(r'src="(.+?)"', file)
    for link in srcLinks:
        urls.add(link)

    hrefLinks = re.findall(r'href="(.+?)"', file)
    for link in hrefLinks:
        urls.add(link)

    httpsLinks = re.findall(r'(^|\s)(https://.+?)($|\s)', file, re.MULTILINE)
    for link in httpsLinks:
        link = link[1].removesuffix('.')
        urls.add(link)

    return urls


def replaceURLInFile(file: str, oldUrl: str, newUrl: str) -> str:
    foundIdx = file.find(oldUrl)
    while foundIdx != -1:
        try:
            needReplaceLeft = False
            if file[foundIdx-len('href="'):foundIdx] == 'href="':
                needReplaceLeft = True
            elif file[foundIdx-len('src="'):foundIdx] == 'src="':
                needReplaceLeft = True
            elif file[foundIdx-len(']('):foundIdx] == '](':
                needReplaceLeft = True
            elif oldUrl.lower().startswith('https://'):
                needReplaceLeft = True
                newUrl = f'[{newUrl}]({newUrl})'

            needReplaceRight = False
            if file[foundIdx+len(oldUrl)] in ')]}>"\' \\\n.,':
                needReplaceRight = True

            if needReplaceLeft and needReplaceRight:
                file = file[0:foundIdx] + newUrl + file[foundIdx+len(oldUrl):]

        except IndexError:
            pass

        foundIdx = file.find(oldUrl, foundIdx+1)

    return file


def isLocalURL(url: str):
    return not ('://' in url or url.startswith('//'))

def isAnchor(url: str):
    return url.startswith('#')

def isMarkdown(url: str):
    if '#' in url:
        url = url.removesuffix('#' + url.split('#')[-1])
    return url.endswith('.md')


def getAllDocuments() -> dict[str, str]:
    docs = dict()
    files = shared.listfiles(os.path.join(EXT_ROOT_DIRECTORY, 'docs'))
    for file in files:
        if not file.endswith(".md"): continue
        fileName = os.path.basename(file).removesuffix(".md").capitalize()
        docs[fileName] = file
    docs["Readme"] = os.path.join(EXT_ROOT_DIRECTORY, 'README.md')
    return docs
