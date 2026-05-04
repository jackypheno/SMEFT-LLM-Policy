import random

OPERATORS = [
    "lq1", "lq3", "qq1", "qq3",
    "qd1", "qd8", "dd", "uu",
    "lequ1", "lequ3"
]

FLAVORS = [1, 2, 3]

SCALES = [1.0, 2.0, 3.0]

def rand_flavor():
    return f"p{random.choice(FLAVORS)} r{random.choice(FLAVORS)} s{random.choice(FLAVORS)} t{random.choice(FLAVORS)}"

def rand_coef():
    return round(random.uniform(-6, 6), 2)

def make_operator():
    op = random.choice(OPERATORS)
    flav = rand_flavor()
    coef = rand_coef()
    return op, flav, coef

def make_theory():

    n_ops = random.randint(1, 10)
    scale = random.choice(SCALES)

    ops = []

    used = set()

    for _ in range(n_ops):

        op, flav, coef = make_operator()

        key = (op, flav)

        # avoid exact duplicates
        if key in used:
            continue

        used.add(key)

        ops.append(
            f"OP_START CLASS_4F BASE_{op} {flav} COEF_{coef} OP_END"
        )

    theory = []

    theory.append("THEORY_START")
    theory.append(f"NUM_OPS_{len(ops)}")
    theory.append(f"SCALE_{scale}")

    theory.extend(ops)

    theory.append("THEORY_END")

    return " ".join(theory)

def generate_dataset(N=100000, outfile="smeft_train.txt"):

    with open(outfile, "w") as f:

        for i in range(N):

            theory = make_theory()
            f.write(theory + "\n")

            if i % 10000 == 0:
                print("generated", i)

if __name__ == "__main__":

    generate_dataset(N=200000)  # adjust to 50k–200k recommended
