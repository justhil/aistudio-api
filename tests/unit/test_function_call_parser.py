from aistudio_api.domain.models import parse_response_chunk


def test_parse_response_chunk_keeps_raw_function_call_and_response():
    chunk = [
        [
            [
                [
                    [
                        [None, None, None, ["getWeather", '{"city":"Shanghai"}']],
                        [None, None, None, None, ["getWeather", {"city": "Shanghai", "temperature": "24C"}]],
                    ]
                ],
                1,
            ]
        ],
        None,
        [5, 1, 6],
        None,
        None,
        None,
        None,
        "resp_123",
    ]

    candidate = parse_response_chunk(chunk)

    assert candidate.function_calls == [
        {
            "type": "functionCall",
            "raw": ["getWeather", '{"city":"Shanghai"}'],
            "name": "getWeather",
            "args": {"city": "Shanghai"},
        }
    ]
    assert candidate.function_responses == [
        {
            "type": "functionResponse",
            "raw": ["getWeather", {"city": "Shanghai", "temperature": "24C"}],
            "name": "getWeather",
            "args": {"city": "Shanghai", "temperature": "24C"},
        }
    ]


def test_parse_response_chunk_extracts_real_aistudio_function_call_shape():
    chunk = [
        [
            [
                [
                    [
                        [
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            [
                                "getWeather",
                                [[["city", [None, None, "Shanghai"]]]],
                                "e6ni61kr",
                            ],
                            None,
                            None,
                            None,
                            "EiYKJGUyNDgzMGE3LTVjZDYtNDJmZS05OThiLWVlNTM5ZTcyYjljMw==",
                        ]
                    ],
                    "model",
                ]
            ]
        ],
        None,
        [52, 15, 147, None, [[1, 52]], None, None, None, None, 80],
        None,
        None,
        None,
        None,
        "resp_real",
    ]

    candidate = parse_response_chunk(chunk)

    assert candidate.function_calls == [
        {
            "type": "functionCall",
            "raw": ["getWeather", [[["city", [None, None, "Shanghai"]]]], "e6ni61kr"],
            "name": "getWeather",
            "args": {"city": "Shanghai"},
            "call_id": "e6ni61kr",
            "thought_signature": "EiYKJGUyNDgzMGE3LTVjZDYtNDJmZS05OThiLWVlNTM5ZTcyYjljMw==",
        }
    ]


def test_parse_response_chunk_decodes_numeric_and_bool_function_call_args():
    # google.protobuf.Value slots: number_value=index1, string_value=index2,
    # bool_value=index3, list_value=index5. Previously only strings decoded.
    chunk = [
        [
            [
                [
                    [
                        [
                            None, None, None, None, None, None, None, None, None, None,
                            [
                                "read",
                                [[
                                    ["limit", [None, 100]],
                                    ["offset", [None, 0]],
                                    ["recursive", [None, None, None, True]],
                                    ["path", [None, None, "README.md"]],
                                    ["lines", [None, None, None, None, None, [[[None, 1], [None, 2]]]]],
                                ]],
                                "call_xyz",
                            ],
                        ]
                    ],
                    "model",
                ]
            ]
        ],
        None,
        [10, 5, 15],
        None, None, None, None,
        "resp_num",
    ]

    candidate = parse_response_chunk(chunk)
    assert candidate.function_calls[0]["name"] == "read"
    assert candidate.function_calls[0]["args"] == {
        "limit": 100,
        "offset": 0,
        "recursive": True,
        "path": "README.md",
        "lines": [1, 2],
    }
