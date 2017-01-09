## TensorFlow version

Since version 0.12 is not necessary compile the library from source on MacOS. This is also for linux version but with an additional dependency.

To build the op for the official binary package simply use `make`, otherwise use the command `make OFFICIAL_BINARY=false`.

## TIPS

Search for symlinks:

```
# search
objdump -t DENN/DENNOp.so
# filter
objdump -t DENN/DENNOp.so | grep CheckOpMessageBuilder
# demangling
objdump -t DENN/DENNOp.so | grep CheckOpMessageBuilder | c++filt
```

## Build notes

* `C_FLAGS += -lprotobuf` is necessary because of `::tensorflow::protobuf::TextFormat::ParseFromString`, otherwise some functions will not be available

## Example benchmark test

```python
datasets_data = [
    (
        "../../../minimal_dataset/data/bezdekIris",
        'load_iris_data',
        ENDict(
            [
                ('name', 'iris_dataset'),
                ('GEN_STEP', 50),
                ('N_GEN', 8),
                ('NP', 100),
                ('BATCH', 40),
                ('W', 0.3),
                ('CR', 0.552),
                ('levels', [
                    # 1 lvl
                    ([4, 3], [3])

                    # 2 lvl
                    # ([4, 8], [8]),
                    # ([8, 3], [3])

                    # 3 lvl
                    # ([4, 8], [8]),
                    # ([8, 8], [8]),
                    # ([8, 3], [3])

                    # 5 lvl
                    # ([4, 8], [8]),
                    # ([8, 8], [8]),
                    # ([8, 8], [8]),
                    # ([8, 8], [8]),
                    # ([8, 3], [3])
                ]),
                ('de_types', [
                    'rand/1/bin',
                    'rand/1/exp',
                    'rand/2/bin',
                    'rand/2/exp'
                ])
            ]
        )
    )
]
```