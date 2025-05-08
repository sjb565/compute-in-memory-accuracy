import sys, io, os, time, random
import numpy as np
from simulator import CrossSimParameters

def dnn_inference_params(**kwargs):
    """
    Pass parameters using kwargs to allow for a general parameter dict
    to be used. This function should be called before train and sets all
    parameters of the neural core simulator

    If ideal=True, parameters for an ideal, floating-point simulation will
    be returned, ignoring most other settings.

    For the meanings of the various parameters, please see the corresponding
    files in /simulation/parameters/

    :return: params, a parameter object with all the settings

    """
    #######################

    #### load relevant parameters from arg dict
    ideal = kwargs.get("ideal",False)
    core_style = kwargs.get("core_style","BALANCED")
    error_model = kwargs.get("error_model","none")
    alpha_error = kwargs.get("alpha_error",0.0)
    proportional_error = kwargs.get("proportional_error",False)

    noise_model = kwargs.get("noise_model","none")
    alpha_noise = kwargs.get("alpha_noise",0.0)
    proportional_noise = kwargs.get("proportional_noise",False)

    t_drift = kwargs.get("t_drift",0)
    drift_model = kwargs.get("drift_model",None)

    Rp_row = kwargs.get("Rp_row",0)
    Rp_col = kwargs.get("Rp_col",0)
    Rp_row_terminal = kwargs.get("Rp_row_terminal",0)
    Rp_col_terminal = kwargs.get("Rp_col_terminal",0)
    NrowsMax = kwargs.get("NrowsMax",None)
    NcolsMax = kwargs.get("NcolsMax",None)
    weight_bits = kwargs.get("weight_bits",0)
    weight_percentile = kwargs.get("weight_percentile",100)

    adc_bits = kwargs.get("adc_bits",8)
    input_bits = kwargs.get("input_bits",8)
    input_range = kwargs.get("input_range",(0,1))
    adc_range = kwargs.get("adc_range",(0,1))
    adc_type = kwargs.get("adc_type","generic")
    positiveInputsOnly = kwargs.get("positiveInputsOnly",False)
    interleaved_posneg = kwargs.get("interleaved_posneg",False)
    current_from_input = kwargs.get("current_from_input",True)
    selected_rows = kwargs.get("selected_rows","top")
    subtract_current_in_xbar = kwargs.get("subtract_current_in_xbar",True)
    Rmin = kwargs.get("Rmin", 1000)
    Rmax = kwargs.get("Rmax", 10000)
    Vread = kwargs.get("Vread",0.1)
    infinite_on_off_ratio = kwargs.get("infinite_on_off_ratio", True)
    
    Nslices = kwargs.get("Nslices",1)    
    digital_offset = kwargs.get("digital_offset",True)
    adc_range_option = kwargs.get("adc_range_option","CALIBRATED")
    Icol_max = kwargs.get("Icol_max",0)
    digital_bias = kwargs.get("digital_bias",False)

    x_par = kwargs.get("x_par",1)
    y_par = kwargs.get("y_par",1)
    useGPU = kwargs.get("useGPU",False)
    gpu_id = kwargs.get("gpu_id",0)

    profile_xbar_inputs = kwargs.get("profile_xbar_inputs",False)
    profile_adc_inputs = kwargs.get("profile_adc_inputs",False)
    ntest = kwargs.get("ntest",1)

    balanced_style = kwargs.get("balanced_style","ONE_SIDED")
    input_bitslicing = kwargs.get("input_bitslicing",False)
    input_slice_size = kwargs.get("input_slice_size",1)
    adc_per_ibit = kwargs.get("adc_per_ibit",False)
    
    ################  create parameter objects with all core settings
    params = CrossSimParameters()

    ############### Numerical simulation settings
    params.simulation.useGPU = useGPU
    if useGPU:
        params.simulation.gpu_id = gpu_id

    # Parameters for SW packing mode (only used if simulation.disable_fast_matmul = True)
    params.simulation.convolution.x_par = int(x_par)
    params.simulation.convolution.y_par = int(y_par)

    if ideal:
        return params

    ############### Crossbar weight mapping settings

    if Nslices == 1:
        params.core.style = core_style
    else:
        params.core.style = "BITSLICED"
        params.core.bit_sliced.style = core_style

    if NrowsMax is not None:
        params.core.rows_max = NrowsMax

    if NcolsMax is not None:
        params.core.cols_max = NcolsMax

    params.core.balanced.style = balanced_style
    params.core.balanced.subtract_current_in_xbar = subtract_current_in_xbar
    if digital_offset:
        params.core.offset.style = "DIGITAL_OFFSET"
    else:
        params.core.offset.style = "UNIT_COLUMN_SUBTRACTION"

    params.xbar.device.Rmin = Rmin
    params.xbar.device.Rmax = Rmax
    params.xbar.device.infinite_on_off_ratio = infinite_on_off_ratio
    params.xbar.device.Vread = Vread

    ############### Device errors

    #### Read noise
    if noise_model == "generic" and alpha_noise > 0:
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.magnitude = alpha_noise
        if not proportional_noise:
            params.xbar.device.read_noise.model = "NormalIndependentDevice"
        else:
            params.xbar.device.read_noise.model = "NormalProportionalDevice"
    elif noise_model != "generic" and noise_model != "none":
        params.xbar.device.read_noise.enable = True
        params.xbar.device.read_noise.model = noise_model

    ##### Programming error
    if error_model == "generic" and alpha_error > 0:
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.magnitude = alpha_error
        if not proportional_error:
            params.xbar.device.programming_error.model = "NormalIndependentDevice"
        else:
            params.xbar.device.programming_error.model = "NormalProportionalDevice"
    elif error_model != "generic" and error_model != "none":
        params.xbar.device.programming_error.enable = True
        params.xbar.device.programming_error.model = error_model

    # Drift
    if drift_model != "none":
        params.xbar.device.drift_error.enable = True
        params.xbar.device.time = t_drift
        params.xbar.device.drift_error.model = drift_model

    ############### Parasitic resistance

    if Rp_col > 0 or Rp_row > 0 or Rp_col_terminal > 0 and Rp_row_terminal > 0:
        # Bit line parasitic resistance
        params.xbar.array.parasitics.enable = True
        params.xbar.array.parasitics.Rp_col = Rp_col
        params.xbar.array.parasitics.Rp_row = Rp_row
        params.xbar.array.parasitics.Rp_col_terminal = Rp_col_terminal
        params.xbar.array.parasitics.Rp_row_terminal = Rp_row_terminal
        params.xbar.array.parasitics.current_from_input = current_from_input
        params.xbar.array.parasitics.selected_rows = selected_rows

        # Numeric params related to parasitic resistance simulation
        params.simulation.Niters_max_parasitics = 100
        params.simulation.Verr_th_mvm = 2e-4
        params.simulation.relaxation_gamma = 0.9 # under-relaxation
        params.simulation.Verr_matmul_criterion = "max_mean" # max over array, max over MVMs

    ############### Weight bit slicing

    # Compute the number of cell bits
    if core_style == "OFFSET" and Nslices == 1:
        cell_bits = weight_bits
    elif core_style == "BALANCED" and Nslices == 1:
        cell_bits = weight_bits - 1
    elif Nslices > 1:
        # For weight bit slicing, quantization is done during mapping and does
        # not need to be applied at the xbar level
        if weight_bits % Nslices == 0:
            cell_bits = int(weight_bits / Nslices)
        elif core_style == "BALANCED":
            cell_bits = int(np.ceil((weight_bits-1)/Nslices))
        else:
            cell_bits = int(np.ceil(weight_bits/Nslices))
        params.core.bit_sliced.num_slices = Nslices

    # Weights
    params.core.weight_bits = int(weight_bits)
    params.core.mapping.weights.percentile = weight_percentile/100

    params.xbar.device.cell_bits = cell_bits
    params.xbar.adc.mvm.adc_per_ibit = (adc_per_ibit and input_bitslicing)
    params.xbar.adc.mvm.adc_range_option = adc_range_option
    params.core.balanced.interleaved_posneg = interleaved_posneg
    params.xbar.array.Icol_max = Icol_max

    ###################### Analytics

    params.simulation.analytics.profile_xbar_inputs = profile_xbar_inputs
    params.simulation.analytics.profile_adc_inputs = profile_adc_inputs
    params.simulation.analytics.ntest = ntest

    ###################### DAC settings

    params.xbar.dac.mvm.bits = int(input_bits)
    if positiveInputsOnly:
        params.xbar.dac.mvm.model = "QuantizerDAC"
        params.xbar.dac.mvm.signed = False
    else:
        params.xbar.dac.mvm.model = "SignMagnitudeDAC"
        params.xbar.dac.mvm.signed = True

    if input_bits > 0:
        params.xbar.dac.mvm.input_bitslicing = input_bitslicing
        params.xbar.dac.mvm.slice_size = input_slice_size
        if not digital_bias:
            input_range[1] = np.maximum(input_range[1],1)

        if not positiveInputsOnly:        
            if input_bitslicing or params.xbar.dac.mvm.model == "SignMagnitudeDAC":
                max_input_range = float(np.max(np.abs(input_range)))
                params.core.mapping.inputs.mvm.min = -max_input_range
                params.core.mapping.inputs.mvm.max = max_input_range
            else:
                params.core.mapping.inputs.mvm.min = float(input_range[0])
                params.core.mapping.inputs.mvm.max = float(input_range[1])
        else:
            params.core.mapping.inputs.mvm.min = float(input_range[0])
            params.core.mapping.inputs.mvm.max = float(input_range[1])
    else:
        params.xbar.dac.mvm.input_bitslicing = False
        params.core.mapping.inputs.mvm.min = -1e10
        params.core.mapping.inputs.mvm.max = 1e10

    ###################### ADC settings

    params.xbar.adc.mvm.bits = int(adc_bits)

    # Custom ADCs are currently set to ideal. Modify the parameters
    # below to model non-ideal ADC implementations
    if adc_type == "ramp":
        params.xbar.adc.mvm.model = "RampADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.symmetric_cdac = True
    elif adc_type == "sar" or adc_type == "SAR":
        params.xbar.adc.mvm.model = "SarADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_capacitor = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.split_cdac = True
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "pipeline":
        params.xbar.adc.mvm.model = "PipelineADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8
    elif adc_type == "cyclic":
        params.xbar.adc.mvm.model = "CyclicADC"
        params.xbar.adc.mvm.gain_db = 100
        params.xbar.adc.mvm.sigma_C1 = 0.00
        params.xbar.adc.mvm.sigma_C2 = 0.00
        params.xbar.adc.mvm.sigma_Cpar = 0.00
        params.xbar.adc.mvm.sigma_comparator = 0.00
        params.xbar.adc.mvm.group_size = 8

    # Determine if signed ADC
    if adc_range_option == "CALIBRATED" and adc_bits > 0:
        params.xbar.adc.mvm.signed = bool(np.min(adc_range) < 0)
    else:
        params.xbar.adc.mvm.signed = (core_style == "BALANCED" or (core_style == "OFFSET" and not positiveInputsOnly))

    # Set the ADC model and the calibrated range
    if adc_bits > 0:
        if Nslices > 1:
            if adc_range_option == "CALIBRATED":
                params.xbar.adc.mvm.calibrated_range = adc_range
            if params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "SignMagnitudeADC"
            elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                params.xbar.adc.mvm.model = "QuantizerADC"
        else:
            # This criterion checks if the center point of the range is within 1 level of zero
            # If that is the case, the range is made symmetric about zero and sign bit is used
            if adc_range_option == "CALIBRATED":
                if np.abs(0.5*(adc_range[0] + adc_range[1])/(adc_range[1] - adc_range[0])) < 1/pow(2,adc_bits):
                    absmax = np.max(np.abs(adc_range))
                    params.xbar.adc.mvm.calibrated_range = np.array([-absmax,absmax])
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "SignMagnitudeADC"
                else:
                    params.xbar.adc.mvm.calibrated_range = adc_range
                    if adc_type == "generic":
                        params.xbar.adc.mvm.model = "QuantizerADC"

            elif adc_range_option == "MAX" or adc_range_option == "GRANULAR":
                if params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "SignMagnitudeADC"
                elif not params.xbar.adc.mvm.signed and adc_type == "generic":
                    params.xbar.adc.mvm.model = "QuantizerADC"
    
    params.validate()

    return params
