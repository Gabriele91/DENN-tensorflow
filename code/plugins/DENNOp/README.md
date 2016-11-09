## Build notes

* `C_FLAGS += -lprotobuf` is necessary because of `::tensorflow::protobuf::TextFormat::ParseFromString`, otherwise some functions will not be available