def prepare_settings(settings):
    return {
        'mean': settings.mean,
        'std': settings.std,
        'url': settings.url,
        'input_range': (0, 1),
        'input_space': 'RGB',
    }
