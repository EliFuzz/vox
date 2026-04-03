use objc2::rc::autoreleasepool;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AnyThread as _;
use objc2_core_ml::{
    MLComputeUnits, MLDictionaryFeatureProvider, MLFeatureProvider, MLFeatureValue, MLModel,
    MLModelConfiguration, MLMultiArray, MLMultiArrayDataType, MLPredictionOptions,
};
use objc2_foundation::{NSDictionary, NSNumber, NSString, NSURL};
use std::path::Path;

pub fn load_model(path: &Path, compute_units: MLComputeUnits) -> Retained<MLModel> {
    let path_str = path.to_str().expect("invalid model path");
    let url = NSURL::fileURLWithPath(&NSString::from_str(path_str));
    let config = unsafe { MLModelConfiguration::new() };
    unsafe { config.setComputeUnits(compute_units) };
    unsafe { MLModel::modelWithContentsOfURL_configuration_error(&url, &config) }
        .expect("failed to load CoreML model")
}

pub fn create_multi_array(
    shape: &[usize],
    data_type: MLMultiArrayDataType,
) -> Retained<MLMultiArray> {
    let ns_shape: Vec<Retained<NSNumber>> = shape
        .iter()
        .map(|&s| NSNumber::new_isize(s as isize))
        .collect();
    let ns_shape_refs: Vec<&NSNumber> = ns_shape.iter().map(|n| &**n).collect();
    let ns_array = objc2_foundation::NSArray::from_slice(&ns_shape_refs);
    unsafe {
        MLMultiArray::initWithShape_dataType_error(MLMultiArray::alloc(), &ns_array, data_type)
    }
    .expect("failed to create MLMultiArray")
}

fn data_ptr<T>(array: &MLMultiArray) -> *mut T {
    #[allow(deprecated)]
    unsafe {
        array.dataPointer().as_ptr() as *mut T
    }
}

pub fn multi_array_f32_ptr(array: &MLMultiArray) -> *mut f32 {
    data_ptr(array)
}

pub fn multi_array_i32_ptr(array: &MLMultiArray) -> *mut i32 {
    data_ptr(array)
}

fn run_prediction(
    model: &MLModel,
    inputs: &NSDictionary<NSString, objc2::runtime::AnyObject>,
    options: Option<&MLPredictionOptions>,
) -> Retained<ProtocolObject<dyn MLFeatureProvider>> {
    let provider = unsafe {
        MLDictionaryFeatureProvider::initWithDictionary_error(
            MLDictionaryFeatureProvider::alloc(),
            inputs,
        )
    }
    .expect("failed to create feature provider");

    let provider_proto: &ProtocolObject<dyn MLFeatureProvider> =
        ProtocolObject::from_ref(&*provider);

    if let Some(opts) = options {
        return unsafe { model.predictionFromFeatures_options_error(provider_proto, opts) }
            .expect("model prediction failed");
    }
    unsafe { model.predictionFromFeatures_error(provider_proto) }.expect("model prediction failed")
}

pub fn predict(
    model: &MLModel,
    inputs: &NSDictionary<NSString, objc2::runtime::AnyObject>,
    options: Option<&MLPredictionOptions>,
) -> Retained<ProtocolObject<dyn MLFeatureProvider>> {
    autoreleasepool(|_| run_prediction(model, inputs, options))
}

pub fn predict_with<T, F>(
    model: &MLModel,
    inputs: &NSDictionary<NSString, objc2::runtime::AnyObject>,
    options: Option<&MLPredictionOptions>,
    extract: F,
) -> T
where
    F: for<'pool> FnOnce(&ProtocolObject<dyn MLFeatureProvider>) -> T,
{
    autoreleasepool(|_| extract(&run_prediction(model, inputs, options)))
}

pub fn feature_value_multi_array(
    provider: &ProtocolObject<dyn MLFeatureProvider>,
    name: &str,
) -> Retained<MLMultiArray> {
    let ns_name = NSString::from_str(name);
    let fv = unsafe { provider.featureValueForName(&ns_name) }
        .unwrap_or_else(|| panic!("missing feature: {name}"));
    unsafe { fv.multiArrayValue() }.unwrap_or_else(|| panic!("feature {name} is not MLMultiArray"))
}

fn ns_number_array_to_vec(
    arr: &objc2_foundation::NSArray<objc2_foundation::NSNumber>,
) -> Vec<isize> {
    (0..arr.len())
        .map(|i| arr.objectAtIndex(i).as_isize())
        .collect()
}

pub fn multi_array_strides(array: &MLMultiArray) -> Vec<isize> {
    ns_number_array_to_vec(unsafe { &array.strides() })
}

pub fn multi_array_shape(array: &MLMultiArray) -> Vec<isize> {
    ns_number_array_to_vec(unsafe { &array.shape() })
}

pub fn make_input_dict(
    entries: &[(&str, &MLMultiArray)],
) -> Retained<NSDictionary<NSString, objc2::runtime::AnyObject>> {
    let keys: Vec<Retained<NSString>> =
        entries.iter().map(|(k, _)| NSString::from_str(k)).collect();
    let vals: Vec<Retained<MLFeatureValue>> = entries
        .iter()
        .map(|(_, arr)| unsafe { MLFeatureValue::featureValueWithMultiArray(arr) })
        .collect();
    let key_refs: Vec<&NSString> = keys.iter().map(|k| &**k).collect();
    let val_refs: Vec<&objc2::runtime::AnyObject> = vals
        .iter()
        .map(|v| {
            let ptr: &objc2::runtime::AnyObject = unsafe { std::mem::transmute(&**v) };
            ptr
        })
        .collect();
    NSDictionary::from_slices(&key_refs, &val_refs)
}
