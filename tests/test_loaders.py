from vca.ingestion.loaders import load_conversation, load_csv, load_json, load_txt


def test_load_json_list():
    conv = load_json('[{"speaker":"customer","text":"I have anxiety"}]')
    assert conv["turns"] == [{"speaker": "customer", "text": "I have anxiety"}]


def test_load_json_object():
    conv = load_json('{"conversation_id":"C9","turns":[{"speaker":"handler","text":"Hi"}]}')
    assert conv["conversation_id"] == "C9"
    assert conv["turns"][0]["speaker"] == "handler"


def test_load_csv_with_header():
    conv = load_csv("speaker,text\ncustomer,I lost my job\nhandler,I am sorry\n")
    assert [t["speaker"] for t in conv["turns"]] == ["customer", "handler"]


def test_load_txt_speaker_prefixes():
    conv = load_txt("Handler: Hello\nCustomer: I can't cope\nJust a bare line")
    speakers = [t["speaker"] for t in conv["turns"]]
    assert speakers == ["handler", "customer", "customer"]  # bare line defaults to customer


def test_dispatch_by_extension_and_sniff():
    assert load_conversation('[{"text":"hi"}]', "x.json")["turns"][0]["text"] == "hi"
    # No extension -> sniff JSON
    assert load_conversation('{"turns":[{"speaker":"customer","text":"hi"}]}', "")["turns"]
    # Plain text sniff
    assert load_conversation("Customer: hello", "")["turns"][0]["speaker"] == "customer"


def test_empty_turns_are_dropped():
    conv = load_csv("customer,\nhandler,Hi there\n")
    assert len(conv["turns"]) == 1
