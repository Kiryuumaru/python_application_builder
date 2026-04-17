import pytest
from application_builder import (
    Configuration,
    ConfigurationSection,
    ConfigurationBuilder,
    MemoryConfigurationProvider,
)


def _build_config(data):
    return ConfigurationBuilder().add_in_memory_collection(data).build()


class TestConfigurationRootProperties:
    def test_key_is_empty_string(self):
        config = _build_config({})
        assert config.key == ""

    def test_path_is_empty_string(self):
        config = _build_config({})
        assert config.path == ""

    def test_value_is_none(self):
        config = _build_config({"key": "val"})
        assert config.value is None


class TestConfigurationGet:
    def test_returns_value_for_existing_key(self):
        config = _build_config({"App:Name": "MyApp"})
        assert config.get("App:Name") == "MyApp"

    def test_returns_default_for_missing_key(self):
        config = _build_config({})
        assert config.get("missing", "fallback") == "fallback"

    def test_returns_none_when_no_default(self):
        config = _build_config({})
        assert config.get("missing") is None


class TestConfigurationGetInt:
    def test_valid_int_string(self):
        config = _build_config({"Port": "8080"})
        assert config.get_int("Port") == 8080

    def test_invalid_string_returns_default(self):
        config = _build_config({"Port": "abc"})
        assert config.get_int("Port", 3000) == 3000

    def test_missing_key_returns_default(self):
        config = _build_config({})
        assert config.get_int("Port", 5000) == 5000

    def test_missing_key_returns_none_without_default(self):
        config = _build_config({})
        assert config.get_int("Port") is None

    def test_negative_int(self):
        config = _build_config({"Offset": "-10"})
        assert config.get_int("Offset") == -10


class TestConfigurationGetFloat:
    def test_valid_float_string(self):
        config = _build_config({"Rate": "3.14"})
        assert config.get_float("Rate") == pytest.approx(3.14)

    def test_integer_string_as_float(self):
        config = _build_config({"Rate": "7"})
        assert config.get_float("Rate") == 7.0

    def test_invalid_string_returns_default(self):
        config = _build_config({"Rate": "not_a_number"})
        assert config.get_float("Rate", 1.0) == 1.0

    def test_missing_key_returns_none_without_default(self):
        config = _build_config({})
        assert config.get_float("Rate") is None


class TestConfigurationGetBool:
    @pytest.mark.parametrize("raw", ["true", "True", "TRUE", "yes", "Yes", "1", "on", "ON"])
    def test_truthy_values(self, raw):
        config = _build_config({"Flag": raw})
        assert config.get_bool("Flag") is True

    @pytest.mark.parametrize("raw", ["false", "False", "FALSE", "no", "No", "0", "off", "OFF"])
    def test_falsy_values(self, raw):
        config = _build_config({"Flag": raw})
        assert config.get_bool("Flag") is False

    def test_invalid_string_returns_default(self):
        config = _build_config({"Flag": "maybe"})
        assert config.get_bool("Flag", True) is True

    def test_missing_key_returns_default(self):
        config = _build_config({})
        assert config.get_bool("Flag", False) is False

    def test_missing_key_returns_none_without_default(self):
        config = _build_config({})
        assert config.get_bool("Flag") is None


class TestConfigurationGetDict:
    def test_valid_json_dict(self):
        config = _build_config({"Data": '{"a": 1, "b": 2}'})
        assert config.get_dict("Data") == {"a": 1, "b": 2}

    def test_invalid_json_returns_default(self):
        config = _build_config({"Data": "not json"})
        assert config.get_dict("Data", {"x": 0}) == {"x": 0}

    def test_missing_key_returns_default(self):
        config = _build_config({})
        assert config.get_dict("Data", {"default": True}) == {"default": True}

    def test_missing_key_returns_none_without_default(self):
        config = _build_config({})
        assert config.get_dict("Data") is None


class TestConfigurationGetList:
    def test_valid_json_array(self):
        config = _build_config({"Items": '[1, 2, 3]'})
        assert config.get_list("Items") == [1, 2, 3]

    def test_comma_separated_fallback(self):
        config = _build_config({"Items": "a, b, c"})
        assert config.get_list("Items") == ["a", "b", "c"]

    def test_comma_separated_strips_whitespace(self):
        config = _build_config({"Items": " x , y , z "})
        result = config.get_list("Items")
        assert result == ["x", "y", "z"]

    def test_invalid_returns_default(self):
        config = _build_config({})
        assert config.get_list("Items", ["default"]) == ["default"]

    def test_missing_key_returns_none_without_default(self):
        config = _build_config({})
        assert config.get_list("Items") is None


class TestConfigurationReload:
    def test_reload_picks_up_new_data(self):
        provider = MemoryConfigurationProvider({"Key": "old"})
        config = Configuration([provider])
        assert config.get("Key") == "old"

        provider.set("Key", "new")
        config.reload()
        assert config.get("Key") == "new"

    def test_reload_picks_up_added_keys(self):
        provider = MemoryConfigurationProvider({})
        config = Configuration([provider])
        assert config.get("New") is None

        provider.set("New", "value")
        config.reload()
        assert config.get("New") == "value"


class TestConfigurationSectionProperties:
    def test_key_property(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        assert section.key == "App"

    def test_path_property(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        assert section.path == "App"

    def test_value_delegates_to_config(self):
        config = _build_config({"App": "root_value"})
        section = config.get_section("App")
        assert section.value == "root_value"

    def test_value_is_none_when_no_direct_value(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        assert section.value is None

    def test_nested_get_section(self):
        config = _build_config({"App:Database:Host": "localhost"})
        section = config.get_section("App").get_section("Database")
        assert section.key == "Database"
        assert section.path == "App:Database"
        assert section.get("Host") == "localhost"

    def test_get_section_empty_key_returns_self(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        same = section.get_section("")
        assert same is section


class TestConfigurationSectionGet:
    def test_get_with_key(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        assert section.get("Name") == "MyApp"

    def test_get_with_empty_key_returns_value(self):
        config = _build_config({"App": "root_val"})
        section = config.get_section("App")
        assert section.get("") == "root_val"

    def test_get_with_empty_key_returns_default_when_no_value(self):
        config = _build_config({"App:Name": "MyApp"})
        section = config.get_section("App")
        assert section.get("", "fallback") == "fallback"

    def test_get_missing_key_returns_default(self):
        config = _build_config({})
        section = config.get_section("App")
        assert section.get("Missing", "default") == "default"


class TestConfigurationSectionGetChildren:
    def test_returns_immediate_children(self):
        config = _build_config({
            "App:Database:Host": "localhost",
            "App:Database:Port": "5432",
            "App:Logging:Level": "Info",
        })
        section = config.get_section("App")
        children = section.get_children()
        keys = [c.key for c in children]
        assert "Database" in keys
        assert "Logging" in keys

    def test_deduplicates_child_keys(self):
        config = _build_config({
            "App:Database:Host": "localhost",
            "App:Database:Port": "5432",
        })
        section = config.get_section("App")
        children = section.get_children()
        keys = [c.key for c in children]
        assert keys.count("Database") == 1

    def test_children_have_correct_path(self):
        config = _build_config({"App:Cache:TTL": "60"})
        section = config.get_section("App")
        children = section.get_children()
        assert len(children) == 1
        assert children[0].path == "App:Cache"


class TestConfigurationSectionTypedAccessors:
    def test_get_int(self):
        config = _build_config({"App:Port": "9090"})
        section = config.get_section("App")
        assert section.get_int("Port") == 9090

    def test_get_int_invalid_returns_default(self):
        config = _build_config({"App:Port": "abc"})
        section = config.get_section("App")
        assert section.get_int("Port", 80) == 80

    def test_get_float(self):
        config = _build_config({"App:Rate": "2.5"})
        section = config.get_section("App")
        assert section.get_float("Rate") == pytest.approx(2.5)

    def test_get_float_invalid_returns_default(self):
        config = _build_config({"App:Rate": "bad"})
        section = config.get_section("App")
        assert section.get_float("Rate", 0.0) == 0.0

    def test_get_bool_true(self):
        config = _build_config({"App:Enabled": "yes"})
        section = config.get_section("App")
        assert section.get_bool("Enabled") is True

    def test_get_bool_false(self):
        config = _build_config({"App:Enabled": "off"})
        section = config.get_section("App")
        assert section.get_bool("Enabled") is False

    def test_get_bool_invalid_returns_default(self):
        config = _build_config({"App:Enabled": "dunno"})
        section = config.get_section("App")
        assert section.get_bool("Enabled", False) is False

    def test_get_dict(self):
        config = _build_config({"App:Meta": '{"env": "prod"}'})
        section = config.get_section("App")
        assert section.get_dict("Meta") == {"env": "prod"}

    def test_get_dict_invalid_returns_default(self):
        config = _build_config({"App:Meta": "nope"})
        section = config.get_section("App")
        assert section.get_dict("Meta", {}) == {}

    def test_get_list_json(self):
        config = _build_config({"App:Tags": '["a", "b"]'})
        section = config.get_section("App")
        assert section.get_list("Tags") == ["a", "b"]

    def test_get_list_comma_separated(self):
        config = _build_config({"App:Tags": "x, y, z"})
        section = config.get_section("App")
        assert section.get_list("Tags") == ["x", "y", "z"]

    def test_get_list_missing_returns_default(self):
        config = _build_config({})
        section = config.get_section("App")
        assert section.get_list("Tags", []) == []


class TestConfigurationGetChildren:
    def test_returns_top_level_sections(self):
        config = _build_config({
            "App:Name": "MyApp",
            "Logging:Level": "Debug",
            "Database:Host": "localhost",
        })
        children = config.get_children()
        keys = [c.key for c in children]
        assert "App" in keys
        assert "Logging" in keys
        assert "Database" in keys

    def test_deduplicates_top_level_keys(self):
        config = _build_config({
            "App:Name": "MyApp",
            "App:Version": "1.0",
        })
        children = config.get_children()
        keys = [c.key for c in children]
        assert keys.count("App") == 1

    def test_flat_keys_appear_as_children(self):
        config = _build_config({"SimpleKey": "value"})
        children = config.get_children()
        assert len(children) == 1
        assert children[0].key == "SimpleKey"


class TestMultipleProvidersOverride:
    def test_later_provider_overrides_earlier(self):
        provider1 = MemoryConfigurationProvider({"Key": "first"})
        provider2 = MemoryConfigurationProvider({"Key": "second"})
        config = Configuration([provider1, provider2])
        assert config.get("Key") == "second"

    def test_earlier_keys_preserved_when_not_overridden(self):
        provider1 = MemoryConfigurationProvider({"A": "1", "B": "2"})
        provider2 = MemoryConfigurationProvider({"B": "override"})
        config = Configuration([provider1, provider2])
        assert config.get("A") == "1"
        assert config.get("B") == "override"

    def test_builder_chained_in_memory_collections(self):
        config = (
            ConfigurationBuilder()
            .add_in_memory_collection({"X": "first"})
            .add_in_memory_collection({"X": "last"})
            .build()
        )
        assert config.get("X") == "last"
