from types import SimpleNamespace

from signalforge.cli.answer_query import print_chunk_summary


def test_print_chunk_summary_uses_document_labels(capsys):
    print_chunk_summary(
        [
            SimpleNamespace(
                chunk_source="document",
                label='[1] NVIDIA Blog, 2026-03-12, "AI Infrastructure Update"',
                score=0.8062,
                url="https://blogs.nvidia.com/blog/ai-infrastructure/",
                accession_number=None,
            )
        ]
    )

    output = capsys.readouterr().out

    assert 'NVIDIA Blog, 2026-03-12, "AI Infrastructure Update"' in output
    assert "url=https://blogs.nvidia.com/blog/ai-infrastructure/" in output
    assert "Item None" not in output
    assert "accession=None" not in output
