def get_network():
    network = [
        dict(type='conv2d', size=64),
        dict(type='flatten'),
        dict(type='dense', size=32, activation='relu'),
        # dict(type='lstm', size=64)
    ]
    return network