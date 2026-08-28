import unittest

from hass_api import (
    HomeAssistant,
    SatelliteInfo,
    _apply_satellite_registry_info,
    _get_entity_filter_domains,
    _get_select_options,
    _multiple_property,
    _number_property,
    _select_property,
)


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.sent = []

    async def send_json(self, message):
        self.sent.append(message)

    async def receive_json(self):
        return next(self.responses)


class SatelliteRegistryTests(unittest.TestCase):
    def test_direct_entity_area_keeps_satellite_object(self):
        satellite = SatelliteInfo(entity_id="assist_satellite.kitchen")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": "kitchen", "device_id": "device-1"},
            {"device-1": {"area_id": "office"}},
        )

        self.assertIsInstance(satellite, SatelliteInfo)
        self.assertEqual("kitchen", satellite.area_id)
        self.assertEqual("device-1", satellite.device_id)

    def test_derived_device_id_is_used_for_area(self):
        satellite = SatelliteInfo(entity_id="assist_satellite.office")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": None, "device_id": "device-2"},
            {"device-2": {"area_id": "office"}},
        )

        self.assertEqual("device-2", satellite.device_id)
        self.assertEqual("office", satellite.area_id)

    def test_input_device_survives_empty_registry_device(self):
        satellite = SatelliteInfo(device_id="device-3")

        _apply_satellite_registry_info(
            satellite,
            {"area_id": None, "device_id": None},
            {"device-3": {"area_id": "bedroom"}},
        )

        self.assertEqual("device-3", satellite.device_id)
        self.assertEqual("bedroom", satellite.area_id)


class EntityFilterTests(unittest.TestCase):
    def test_current_filter_object(self):
        self.assertEqual(
            {"light"},
            _get_entity_filter_domains({"filter": {"domain": "light"}}),
        )

    def test_legacy_filter_list(self):
        self.assertEqual(
            {"light", "switch"},
            _get_entity_filter_domains({"filter": [{"domain": ["light", "switch"]}]}),
        )

    def test_filter_alternatives_combine_domains(self):
        self.assertEqual(
            {"light", "switch"},
            _get_entity_filter_domains(
                {"filter": [{"domain": "light"}, {"domain": "switch"}]}
            ),
        )

    def test_unrestricted_filter_alternative_does_not_restrict_domains(self):
        self.assertIsNone(
            _get_entity_filter_domains(
                {"filter": [{"domain": "light"}, {"device_class": "outlet"}]}
            )
        )


class SelectorSchemaTests(unittest.TestCase):
    def test_labelled_select_options_use_submitted_values(self):
        self.assertEqual(
            ["low", "high", "auto"],
            _get_select_options(
                {
                    "options": [
                        {"label": "Low speed", "value": "low"},
                        {"label": "High speed", "value": "high"},
                        "auto",
                    ]
                }
            ),
        )

    def test_multiple_selector_wraps_item_schema(self):
        self.assertEqual(
            {
                "type": "array",
                "items": {"type": "string", "enum": ["Kitchen", "Office"]},
                "uniqueItems": True,
            },
            _multiple_property({"type": "string", "enum": ["Kitchen", "Office"]}),
        )

    def test_number_step_is_not_a_runtime_constraint(self):
        self.assertEqual(
            {"type": "number", "minimum": 0.5, "maximum": 10},
            _number_property({"min": 0.5, "max": 10, "step": 1}),
        )

    def test_custom_select_accepts_values_outside_suggestions(self):
        self.assertEqual(
            {"type": "string"},
            _select_property({"options": ["suggestion"], "custom_value": True}),
        )


class ScriptConfigTests(unittest.IsolatedAsyncioTestCase):
    async def test_loads_description_and_fields_from_script_config(self):
        websocket = FakeWebSocket(
            [
                {
                    "success": True,
                    "result": {
                        "config": {
                            "alias": "Notify Mike (Test)",
                            "description": "Notify Mike with a message",
                            "fields": {"message": {"required": True}},
                        }
                    },
                }
            ]
        )
        hass = HomeAssistant(token="token")
        current_id = 0

        def next_id():
            nonlocal current_id
            current_id += 1
            return current_id

        configs = await hass._get_script_configs(websocket, next_id, {"notify_mike"})

        self.assertEqual(
            "Notify Mike with a message",
            configs["notify_mike"]["description"],
        )
        self.assertEqual(
            [
                {
                    "id": 1,
                    "type": "script/config",
                    "entity_id": "script.notify_mike",
                }
            ],
            websocket.sent,
        )

    async def test_stops_when_script_config_command_is_unsupported(self):
        websocket = FakeWebSocket(
            [
                {
                    "success": False,
                    "error": {"code": "unknown_command"},
                }
            ]
        )
        hass = HomeAssistant(token="token")

        configs = await hass._get_script_configs(
            websocket, iter(range(1, 10)).__next__, {"one", "two"}
        )

        self.assertEqual({}, configs)
        self.assertEqual(1, len(websocket.sent))


if __name__ == "__main__":
    unittest.main()
