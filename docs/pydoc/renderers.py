import sys
import io
import dataclasses
import docspec
import typing as t
from pathlib import Path
from pydoc_markdown.interfaces import Context, Renderer, Resolver
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
import html

README_FRONTMATTER = """---
title: {title}
excerpt: {excerpt}
category: {category}
slug: {slug}
order: {order}
hidden: false
---

"""


class HaystackMarkdownRenderer(MarkdownRenderer):
    """
    Custom Markdown renderer heavily based on the `MarkdownRenderer`
    """

    def _render_object(self, fp, level, obj):
        """
        This is where docstrings for a certain object are processed,
        we need to override it in order to better manage new lines.
        """
        super()._render_object(fp, level, obj)


@dataclasses.dataclass
class ReadmeRenderer(Renderer):
    """
    This custom Renderer is heavily based on the `MarkdownRenderer`,
    it just prepends a front matter so that the output can be published
    directly to readme.io.
    """

    # These settings will be used in the front matter output
    title: str
    category: str
    excerpt: str
    slug: str
    order: int
    # This exposes a special `markdown` settings value that can be used to pass
    # parameters to the underlying `MarkdownRenderer`
    markdown: HaystackMarkdownRenderer = dataclasses.field(default_factory=HaystackMarkdownRenderer)

    def init(self, context: Context) -> None:
        self.markdown.init(context)

    def render(self, modules: t.List[docspec.Module]) -> None:
        if self.markdown.filename is None:
            sys.stdout.write(self._frontmatter())
            self.markdown.render_to_stream(modules, sys.stdout)
        else:
            with io.open(self.markdown.filename, "w", encoding=self.markdown.encoding) as fp:
                fp.write(self._frontmatter())
                self.markdown.render_to_stream(modules, t.cast(t.TextIO, fp))

    def _frontmatter(self) -> str:
        return README_FRONTMATTER.format(
            title=self.title, category=self.category, excerpt=self.excerpt, slug=self.slug, order=self.order
        )
