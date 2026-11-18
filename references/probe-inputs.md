# Phase 0 probe input

Supply an existing UTF-8 Markdown file. English paragraphs, headings, inline `$...$` and display
`$$...$$` are tested. See `examples/phase0/common.md` for fractions, scripts, roots, integrals,
sums, limits, vectors, Greek symbols, units, statistics and a matrix.

This is a tested corpus, **not a complete LaTeX whitelist**. Production whitelist and full fallback
auditing are deferred. Handnote coverage's `unknown_commands` is advisory: it can label valid
MathText commands such as `sin` or `lim` unknown. The actual renderer's formula failure warnings
are fatal in the adapter.

Handnote treats a standalone `---` as a page break. MarkdownToHandwrite uses `<newpage>` instead;
the probe preserves source and does not silently translate page-break syntax between them.
Use repeated paragraphs to compare automatic pagination from identical Markdown.

Output directories must be new or empty. No question recognition, OCR, symbolic checking, auto-editing
of equations, layout-repair loop or question/page invariant is implemented in Phase 0.
