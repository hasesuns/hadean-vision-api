from pathlib import Path

# Remove estimation results related to tests in advance.
data_dir_path = Path(__file__).parent.parent.resolve() / "data/tests"
test_result_data_path = data_dir_path / "output"

delete_data_list = []
for extension in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG", "json", "JSON"]:
    delete_data_list += [str(file_path) for file_path in test_result_data_path.glob(f"*.{extension}")]
for delete_data_path in delete_data_list:
    Path(delete_data_path).unlink()
