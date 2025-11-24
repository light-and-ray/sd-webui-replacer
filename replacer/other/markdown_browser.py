import os
import urllib.parse
import gradio as gr
from replacer.other.markdown_browser_tools import ( getURLsFromFile, JS_PREFIX, isLocalURL, isAnchor, isMarkdown,
    replaceURLInFile, getAllDocuments,
)
from replacer.options import EXT_ROOT_DIRECTORY


def renderMarkdownFile(filePath: str, extDir: str):
    with open(filePath, mode='r', encoding="utf-8-sig") as f:
        file = f.read()

    for url in getURLsFromFile(file):
        originalURL = url
        replacementUrl = None
        if JS_PREFIX in originalURL:
            file = file.replace(originalURL, "***")
            continue

        if isLocalURL(url):
            if isAnchor(url): continue
            if '#' in url:
                url = url.removesuffix('#' + url.split('#')[-1])

            if url[0] == '/':
                urlFullPath = os.path.join(extDir, url[1:])
            else:
                urlFullPath = os.path.join(os.path.dirname(filePath), url)

            if os.path.exists(urlFullPath):
                if isMarkdown(url):
                    replacementUrl = f"{JS_PREFIX}markdown_browser_openSubFile('{urlFullPath}')"
                else:
                    replacementUrl = f'file={urlFullPath}'

        if replacementUrl is not None:
            replacementUrl = urllib.parse.quote(replacementUrl)
            file = replaceURLInFile(file, originalURL, replacementUrl)

    return file


def openSubFile(filePath: str):
    file = renderMarkdownFile(filePath, EXT_ROOT_DIRECTORY)
    return file

def openDocument(docName: str):
    if not docName: return ""
    file = renderMarkdownFile(g_documents[docName], EXT_ROOT_DIRECTORY)
    return file


markdownFile = gr.Markdown("", elem_classes=['markdown-browser-file'], elem_id='markdown_browser_file')
g_documents = None

def getDocsTabUI():
    global g_documents
    g_documents = getAllDocuments()

    with gr.Blocks() as tab:
        dummy_component = gr.Textbox("", visible=False)

        with gr.Row():
            selectedDocument = gr.Dropdown(
                label="Document",
                value="",
                choices=[""] + list(g_documents.keys())
            )
            selectButton = gr.Button('Select')
            selectButton.click(
                fn=openDocument,
                inputs=[selectedDocument],
                outputs=[markdownFile],
            ).then(
                fn=None,
                _js='markdown_browser_afterRender',
            )

        with gr.Row():
            markdownFile.render()

        openSubFileButton = gr.Button("", visible=False, elem_id="markdown_browser_openSubFileButton")
        openSubFileButton.click(
            fn=openSubFile,
            _js="markdown_browser_openSubFile_",
            inputs=[dummy_component],
            outputs=[markdownFile]
        ).then(
            fn=None,
            _js='markdown_browser_afterRender',
        )

    return tab

