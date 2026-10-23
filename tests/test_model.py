import numpy as np
import pytest
import torch

from pianist import tokenizer as tok
from pianist.data import Corpus, RandomWindows, eval_windows
from pianist.model import KVCache, ModelConfig, Pianist
from pianist.sampling import generate

CFG = ModelConfig(seq_len=64, d_model=64, n_layers=2, n_heads=4, dropout=0.0)


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Pianist(CFG).eval()


def random_tokens(n, seed=0):
    return torch.randint(3, tok.VOCAB_SIZE, (1, n), generator=torch.Generator().manual_seed(seed))


def test_logits_shape_and_weight_tying(model):
    assert model(random_tokens(10)).shape == (1, 10, tok.VOCAB_SIZE)
    assert model.head.weight is model.embed.weight


def test_attention_is_causal(model):
    x = random_tokens(32)
    y = x.clone()
    y[0, 20:] = random_tokens(12, seed=1)
    torch.testing.assert_close(model(x)[:, :20], model(y)[:, :20])


def test_kv_cache_matches_full_forward(model):
    x = random_tokens(24)
    full = model(x)
    cache = KVCache(CFG, 1, "cpu", torch.float32)
    steps = [model(x[:, :10], cache)]
    steps += [model(x[:, i:i + 1], cache) for i in range(10, 24)]
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-4, rtol=1e-4)


def test_loss_ignores_padding(model):
    x = random_tokens(17)
    padded = torch.cat([x, torch.full((1, 5), tok.PAD)], dim=1)
    torch.testing.assert_close(model.loss(x[:, :-1], x[:, 1:]),
                               model.loss(padded[:, :16], padded[:, 1:17]))


def test_sampling_is_grammatical_and_slides_past_the_context(model):
    tokens = generate(model, max_new_tokens=200, min_notes=40,
                      generator=torch.Generator().manual_seed(0))
    assert len(tokens) > CFG.seq_len  # forced to continue past one context window
    grammar = tok.grammar_table()
    assert all(grammar[a, b] for a, b in zip(tokens[:-1], tokens[1:], strict=True))
    assert len(tok.decode(tokens)) >= min(40, (len(tokens) - 1) // 4)


def test_sampling_is_reproducible(model):
    run = lambda: generate(model, max_new_tokens=50, generator=torch.Generator().manual_seed(3))  # noqa: E731
    assert run() == run()


def small_corpus():
    rng = np.random.default_rng(0)
    pieces = []
    for n in (30, 50, 80):
        onset = np.cumsum(rng.uniform(0, 0.3, n))
        pieces.append(tok.Notes(onset, rng.uniform(0.05, 1, n), rng.integers(40, 90, n),
                                rng.integers(20, 110, n)))
    return Corpus.from_pieces(pieces)


def test_random_windows_have_fixed_length_and_valid_tokens():
    data = RandomWindows(small_corpus(), seq_len=64)
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = data.window(rng)
        assert w.shape == (65,) and w.min() >= 0 and w.max() < tok.VOCAB_SIZE
        real = w[w != tok.PAD]
        grammar = tok.grammar_table()
        assert all(grammar[a, b] for a, b in zip(real[1:-1], real[2:], strict=True))


def test_eval_windows_score_every_token_once():
    corpus = small_corpus()
    windows = eval_windows(corpus, seq_len=64)
    expected = sum(len(tok.encode(corpus.piece(i))) - 1 for i in range(len(corpus)))
    assert int((windows[:, 1:] != tok.PAD).sum()) == expected


def test_corpus_save_load(tmp_path):
    corpus = small_corpus()
    corpus.save(tmp_path / "c.npz")
    back = Corpus.load(tmp_path / "c.npz")
    np.testing.assert_array_equal(back.starts, corpus.starts)
    np.testing.assert_allclose(back.piece(2).onset, corpus.piece(2).onset, rtol=1e-6)


def test_corpus_concat_keeps_piece_boundaries():
    a, b = small_corpus(), small_corpus()
    both = Corpus.concat([a, b])
    assert len(both) == 6 and both.num_notes == 2 * a.num_notes
    np.testing.assert_allclose(both.piece(4).onset, a.piece(1).onset, rtol=1e-6)
    assert both.piece_of(a.num_notes) == 3


def test_encoding_accepts_compact_dtypes():
    corpus = small_corpus()  # stored as float32 / int8
    raw = corpus.notes[: corpus.starts[1]]
    np.testing.assert_array_equal(tok.encode(raw), tok.encode(corpus.piece(0)))


def test_ngram_overlap_flags_the_same_piece_but_not_others():
    from pianist.data import pitch_ngrams

    corpus = small_corpus()
    reference = pitch_ngrams(corpus.piece(2))
    assert np.isin(pitch_ngrams(corpus.piece(2)), reference).mean() == 1.0
    assert np.isin(pitch_ngrams(corpus.piece(1)), reference).mean() < 0.05
