import json
from importlib import resources

import jsonschema
from referencing import Registry, Resource

from training_data_packer.metadata import Metadata

_resource_dir = resources.files("training_data_packer.metadata").joinpath("resources")


def _load_json_resource(name: str) -> dict:
    text = _resource_dir.joinpath(name).read_text()
    return json.loads(text)


class Validator:
    def __init__(self):
        self.registry = Registry()
        for s in ["metadata.json", "part.json", "release-part.json"]:
            schema = _load_json_resource(s)
            self.registry = self.registry.with_resource(schema["$id"], Resource.from_contents(schema))
        self.validator_class = jsonschema.Draft202012Validator

    def _validator(self, schema_url: str, data: dict) -> bool:
        schema = self.registry.resolver().lookup(schema_url).contents
        validator = self.validator_class(schema, registry=self.registry)
        try:
            validator.validate(data)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Validation error for schema {schema_url}: {e}") from e
        return True

    def validate_metadata(self, metadata: Metadata) -> bool:
        return self._validator("https://openeurollm.eu/schemas/metadata.json", metadata.data)

    def validate_release_part(self, data: dict) -> bool:
        return self._validator("https://openeurollm.eu/schemas/release-part.json", data)
