from align_geometry.eval.gsm8k import extract, gold_answer, is_correct, pass_at_k


def test_gold_answer_strips_commas():
    assert gold_answer("She has 3 + 4 = 7 apples.\n#### 1,250") == "1250"


def test_extract_prefers_the_last_box():
    text = "first try \\boxed{12}, no wait, the answer is \\boxed{18}"
    assert extract(text) == "18"


def test_extract_cleans_up_formatting():
    assert extract("so it costs \\boxed{\\$1,200.}") == "1200"
    assert extract("The total is 4,000 dollars.") == "4000"


def test_extract_gives_up_without_numbers():
    assert extract("I don't know") is None


def test_is_correct_compares_numbers_not_strings():
    assert is_correct("18.0", "18")
    assert not is_correct("17", "18")
    assert not is_correct(None, "18")


def test_pass_at_k_edge_cases():
    assert pass_at_k(8, 0, 1) == 0.0
    assert pass_at_k(8, 8, 1) == 1.0
    assert abs(pass_at_k(8, 2, 1) - 0.25) < 1e-12
    # one right answer in eight: two draws without replacement find it 1 - 21/28 of the time
    assert abs(pass_at_k(8, 1, 2) - 0.25) < 1e-12
    assert pass_at_k(8, 1, 8) == 1.0
