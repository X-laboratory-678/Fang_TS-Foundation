from tsfoundation.evaluation.inference import NaiveForecaster, parse_model_response


def test_prediction_parser_accepts_list_text():
    result = parse_model_response("The answer is [1.20, 2.50, 3.75].", horizon=3)
    assert result.predicted_output == [1.2, 2.5, 3.75]


def test_naive_forecaster_output_shape():
    result = NaiveForecaster(horizon=3).predict("[1.0, 2.0, 3.0]")
    assert result.predicted_output == [4.0, 5.0, 6.0]

