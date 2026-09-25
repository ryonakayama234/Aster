"""Text extraction that records changes and does not execute source content."""

import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from docutils import nodes
from docutils.core import publish_doctree
from docutils.parsers.rst import Directive, directives, roles


class Deferred(ValueError):
    pass


@dataclass
class Extracted:
    text: str
    segments: list[dict]
    changes: list[dict]


class MetadataDirective(Directive):
    has_content = True
    optional_arguments = 1
    final_argument_whitespace = True

    def run(self):
        if self.name == 'only' and self.arguments != ['html']:
            raise Deferred('unsupported RST only condition')
        node = nodes.comment(self.block_text, self.block_text)
        node.line = self.lineno
        node['aster_reason'] = 'rst_metadata:' + self.name
        return [node]


class MethodDirective(Directive):
    has_content = True
    required_arguments = 1
    final_argument_whitespace = True
    option_spec = {'noindex': directives.flag}

    def run(self):
        container = nodes.container()
        label = nodes.paragraph(text=self.arguments[0])
        label.line = self.lineno
        container += label
        self.state.nested_parse(self.content, self.content_offset, container)
        return [container]


class SeeAlsoDirective(Directive):
    has_content = True

    def run(self):
        node = nodes.admonition()
        node.line = self.lineno
        node += nodes.title(text='See also')
        self.state.nested_parse(self.content, self.content_offset, node)
        return [node]


def reference_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: object,
    options: Mapping[str, object] | None = None,
    content: Sequence[str] | None = None,
) -> tuple[list[nodes.Node], list[nodes.Node]]:
    # Preserve visible label; an unresolved cross-document target is not fetched.
    del name, lineno, inliner, content
    match = re.fullmatch(r'(.+?)\s*<[^>]+>', text)
    label = match.group(1) if match else text.lstrip('~')
    return [nodes.literal(rawtext, label, **dict(options or {}))], []


def rst_extract(text: str) -> Extracted:
    # These local registrations adapt CPython tutorial markup, not arbitrary Sphinx.
    for name in ['index', 'sectionauthor', 'only']:
        directives.register_directive(name, MetadataDirective)
    directives.register_directive('method', MethodDirective)
    directives.register_directive('seealso', SeeAlsoDirective)
    for name in ['term', 'ref', 'class', 'func', 'meth', 'attr', 'mod', 'exc', 'data',
                 'const', 'keyword', 'token', 'option', 'file', 'pep', 'program', 'dfn', 'kbd', 'samp']:
        roles.register_local_role(name, reference_role)
    warnings = io.StringIO()
    tree = publish_doctree(text, settings_overrides={
        'file_insertion_enabled': False, 'raw_enabled': False,
        'halt_level': 6, 'report_level': 2, 'warning_stream': warnings,
        'syntax_highlight': 'none', '_disable_config': True,
    })
    problems = [n.astext() for n in tree.findall(nodes.system_message) if n['level'] >= 2]
    if problems:
        raise Deferred('RST requires review: ' + problems[0][:250])
    chunks, segments, changes = [], [], []
    offset = 0
    units = (nodes.title, nodes.subtitle, nodes.paragraph, nodes.literal_block,
             nodes.doctest_block, nodes.math_block, nodes.rubric, nodes.term)

    def walk(node):
        nonlocal offset
        if isinstance(node, nodes.comment):
            changes.append({'operation': 'omit_metadata', 'source_line': node.line,
                            'reason': node.get('aster_reason', 'rst_comment')})
            return
        if isinstance(node, (nodes.target, nodes.substitution_definition, nodes.system_message)):
            return
        if isinstance(node, units):
            value = node.astext()
            if not value.strip():
                return
            if isinstance(node, (nodes.literal_block, nodes.doctest_block)):
                value = '```\n' + value + '\n```'
            elif isinstance(node, nodes.paragraph) and isinstance(node.parent, nodes.list_item):
                value = '- ' + value
            chunk = value + '\n\n'
            segments.append({'source_line': node.line, 'node_kind': node.tagname,
                             'output_chars': [offset, offset + len(chunk)]})
            chunks.append(chunk)
            offset += len(chunk)
            return
        for child in node.children:
            if isinstance(child, nodes.Element):
                walk(child)

    walk(tree)
    result = ''.join(chunks)
    if not result.strip():
        raise Deferred('no supported RST text nodes')
    changes.append({'operation': 'render_rst_nodes', 'reason': 'retain prose and code; remove presentation markup'})
    return Extracted(result, segments, changes)


def extract(record: dict) -> Extracted:
    if record['kind'] == 'document':
        text = record['text']
        if record.get('text_format') == 'rst':
            return rst_extract(text)
        if record.get('text_format') != 'plain':
            raise Deferred('unsupported text format')
        # Keep source prose unchanged; do not guess whether an embedded URL is noise.
        return Extracted(text, [{'source_field': 'text', 'output_chars': [0, len(text)]}], [])
    if record['kind'] == 'conversation':
        chunks, segments, changes = [], [], []
        offset = 0
        for number, message in enumerate(record['messages']):
            role = message['role']
            if role not in {'user', 'assistant'}:
                raise Deferred('unsupported conversation role')
            content = message['content']
            if 'Browsing web' in content:
                raise Deferred('mixed execution display requires source-specific annotation')
            chunk = ('User' if role == 'user' else 'Assistant') + ':\n' + content.rstrip('\n') + '\n\n'
            segments.append({'message_index': number, 'source_lines': message.get('source_lines'),
                             'output_chars': [offset, offset + len(chunk)]})
            chunks.append(chunk)
            offset += len(chunk)
        changes.append({'operation': 'serialize_roles', 'reason': 'pretrain text; not SFT or verified tool observations'})
        return Extracted(''.join(chunks), segments, changes)
    raise Deferred('unsupported canonical kind')
