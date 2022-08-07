
from ray.tune.sample import Float, Quantized, LogUniform, Uniform, Integer, Categorical
import warnings
from ax import ChoiceParameter, ParameterType

from pandas.api.types import infer_dtype

def config2ax(tuning):
    def resolve_value(par, domain):
            sampler = domain.get_sampler()
            if isinstance(sampler, Quantized):
                warnings.warn("AxSearch does not support quantization. "
                               "Dropped quantization.")
                sampler = sampler.sampler

            if isinstance(domain, Float):
                if isinstance(sampler, LogUniform):
                    return {
                        "name": par,
                        "type": "range",
                        "bounds": [domain.lower, domain.upper],
                        "value_type": "float",
                        "log_scale": True
                    }
                elif isinstance(sampler, Uniform):
                    return {
                        "name": par,
                        "type": "range",
                        "bounds": [domain.lower, domain.upper],
                        "value_type": "float",
                        "log_scale": False
                    }
            elif isinstance(domain, Integer):
                if isinstance(sampler, LogUniform):
                    return {
                        "name": par,
                        "type": "range",
                        "bounds": [domain.lower, domain.upper - 1],
                        "value_type": "int",
                        "log_scale": True
                    }
                elif isinstance(sampler, Uniform):
                    return {
                        "name": par,
                        "type": "range",
                        "bounds": [domain.lower, domain.upper - 1],
                        "value_type": "int",
                        "log_scale": False
                    }
            elif isinstance(domain, Categorical):
                if isinstance(sampler, Uniform):
                    # _values = domain.categories
                    # _type = infer_dtype(_values, skipna=False)
                    
                    # t2t ={
                    #     'string': ParameterType.STRING,
                    #     'integer':ParameterType.INT,
                    #     'floating':ParameterType.FLOAT,
                    #     'boolean':ParameterType.BOOL
                    # }
                    # if _type not in t2t:
                    #     raise ValueError('AxSearch does not support parameters of type {} in {}'.format(_type, par))
                    # else:
                    #     _type = t2t[_type]
                    # return ChoiceParameter(name=par, values=domain.categories, parameter_type=_type,sort_values=False,is_ordered=False)

                    return {
                            "name": par,
                            "type": "choice",
                            "values": domain.categories,
                            "is_ordered" : False
                        }

            raise ValueError("AxSearch does not support parameters of type "
                             "`{}` with samplers of type `{}`".format(
                                 type(domain).__name__,
                                 type(domain.sampler).__name__))
    
            
    resolved_values = [ resolve_value(path, domain)
            for path, domain in tuning.dict.items()
        ]

    return resolved_values