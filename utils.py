def init_config(config, **kwargs):
    final_params = {}
    for k, v in kwargs.items():
        if hasattr(config, k):
            final_params[k] = v
        elif "." in k:
            # allow --some_config.some_param=True
            config_name, param_name = k.split(".")
            if config.__name__ == config_name:
                if hasattr(config, param_name):
                    final_params[param_name] = v
                else:
                    # In case of specialized config we can warm user
                    print(f"Warning: {config_name} does not accept parameter: {k}")
    
    return config(**final_params)