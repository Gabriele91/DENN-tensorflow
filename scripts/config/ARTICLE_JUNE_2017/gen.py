template_name = "mnist_inheritance_g{0}_d{1}_{2}.json"
template_file = '''
{{
        "name": "MNIST_INHERIT_g{0}_d{1}_{2}",
        "dataset_file": "../datasets/ARTICLE_mnist_minibatch_v004_576x100_1s.gz",
        "TOT_GEN": 20000,
        "GEN_STEP": {0},
        "GEN_SAMPLES": 1,
        "VALIDATION_STEP": 1000,
        "TYPE": "float",
        "F": 1.0,
        "CR": 0.9,
        "NP": 96,
        "training": true,
        "JDE": 0.1,
        "inheritance": {{
            "d": {3},
            "when": "{2}"
        }},
        "de_types": [
            "rand/1/bin"
        ],
        "reset_every": {{
            "counter": 2
        }},
        "clamp": {{
            "min": -5,
            "max": 5
        }},
        "levels": [
            @import(mnist_one_lvl_shape.json)
        ]
}}
'''

all_d = [0.7, 0.8, 0.9]
all_gen_step = [20, 50, 100]
all_when = ['always', 'batch']

for when in all_when:
    for gen_step in all_gen_step:
        for d in all_d:
            file_ = template_name.format(gen_step, str(d).replace(".", ""), when)
            with open(file_, "w") as new_task:
                new_task.write(template_file.format(gen_step, str(d).replace(".", ""), when, d))
            print("@import(ARTICLE_JUNE_2017/{}),".format(file_))
