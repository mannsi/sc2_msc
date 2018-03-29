def get_network():
    network = [
        dict(type='conv2d', size=32),
        dict(type='flatten'),
        dict(type='dense', size=32, activation='relu'),
    ]
    return network
