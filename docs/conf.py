from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

project = 'tpuasm'
author = 'Ayaka Mikazuki'
copyright = '2026, Ayaka Mikazuki'
language = 'zh_CN'
extensions = ['myst_parser', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.intersphinx']
html_theme = 'sphinx_book_theme'
myst_heading_anchors = 3
html_title = 'tpuasm'
html_logo = '_static/tpuasm-logo.png'
html_theme_options = {
    'repository_url': 'https://github.com/ayaka14732/tpuasm',
    'use_repository_button': True,
}

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_use_ivar = True
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'jax': ('https://docs.jax.dev/en/latest/', None),
}

REPOSITORY_URL = 'https://github.com/ayaka14732/tpuasm'
DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
LINK = re.compile(r'\]\(([^)#\s:]+)(#[^)]*)?\)')

def repository_links(app: object, docname: str, source: list[str]) -> None:
    """站点外的仓库文件不属于文档源，把指向它们的相对链接改为 GitHub 链接。"""
    directory = (DOCS / docname).parent

    def replace(match: re.Match[str]) -> str:
        path = (directory / match[1]).resolve()
        if path.is_relative_to(DOCS) or not path.is_relative_to(ROOT):
            return match[0]
        kind = 'tree' if path.is_dir() else 'blob'
        return f']({REPOSITORY_URL}/{kind}/main/{path.relative_to(ROOT).as_posix()}{match[2] or ""})'

    source[0] = LINK.sub(replace, source[0])

def setup(app: object) -> None:
    app.connect('source-read', repository_links)
