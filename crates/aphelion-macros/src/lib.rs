use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Fields, Generics, Item, ItemStruct, Type};

/// Attribute macro for marking a model builder type.
///
/// Generates a `ConfigSpec` impl and a `ModelBuilder` impl
/// that forwards to `build_graph(&self, backend, trace)`.
///
/// # Required Fields
///
/// - `config: ModelConfig` - Must be present on the struct. This field is exposed via the `ConfigSpec` trait.
///
/// # Supported Field Attributes
///
/// The macro supports the following field-level attributes for future use:
///
/// - `#[aphelion(skip)]` - Skip this field from auto-generated ConfigSpec (reserved for future use)
/// - `#[aphelion(rename = "new_name")]` - Rename the field in the generated config (reserved for future use)
///
/// # Required Methods
///
/// The struct must implement the following method:
/// ```text
/// fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph
/// ```
///
/// This method is called by the generated `ModelBuilder::build` implementation.
///
/// # Generic Type Support
///
/// The macro fully supports generic types with any bounds:
///
/// ```text
/// #[aphelion_model]
/// struct GenericModel<T: Clone + Send> {
///     config: ModelConfig,
///     data: T,
/// }
///
/// impl<T: Clone + Send> GenericModel<T> {
///     fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph {
///         // Implementation
///     }
/// }
/// ```
///
/// # Examples
///
/// Basic usage:
/// ```ignore
/// use aphelion_core::{
///     aphelion_model, config::ModelConfig, backend::Backend,
///     diagnostics::TraceSink, graph::BuildGraph,
/// };
///
/// #[aphelion_model]
/// struct MyModel {
///     config: ModelConfig,
///     internal_state: String,
/// }
///
/// impl MyModel {
///     fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph {
///         let mut graph = BuildGraph::default();
///         let node = graph.add_node("my_model", self.config.clone());
///         graph.add_edge(node, node);
///         graph
///     }
/// }
/// ```
///
/// With generic types:
/// ```ignore
/// #[aphelion_model]
/// struct GenericModel<T: Clone + Send> {
///     config: ModelConfig,
///     data: T,
/// }
///
/// impl<T: Clone + Send> GenericModel<T> {
///     fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph {
///         let mut graph = BuildGraph::default();
///         let node = graph.add_node("generic_model", self.config.clone());
///         graph.add_edge(node, node);
///         graph
///     }
/// }
/// ```
///
/// # Errors
///
/// Compile errors will be generated if:
/// - The struct does not have a `config: ModelConfig` field
/// - The `config` field is not of type `ModelConfig`
/// - The struct does not implement a `build_graph` method with the correct signature
///   (caught at usage time in trait impl)
///
/// # Generated Implementations
///
/// The macro generates two trait implementations:
///
/// 1. `ConfigSpec` - Returns a reference to the `config` field
/// 2. `ModelBuilder` - Forwards `build` calls to your `build_graph` method
#[proc_macro_attribute]
pub fn aphelion_model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as Item);
    let output = match input {
        Item::Struct(item_struct) => expand_model_struct(item_struct),
        other => quote! { #other },
    };
    TokenStream::from(output)
}

/// Check if a type is ModelConfig
fn is_model_config(ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            type_path.path.segments.last().map(|s| s.ident == "ModelConfig").unwrap_or(false)
        }
        _ => false,
    }
}

/// Check if a struct has any generic parameters
fn has_generics(generics: &Generics) -> bool {
    !generics.params.is_empty()
}

fn expand_model_struct(item_struct: ItemStruct) -> proc_macro2::TokenStream {
    let name = &item_struct.ident;
    let generics: &Generics = &item_struct.generics;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Check that this is a named struct (not a tuple struct or unit struct)
    match &item_struct.fields {
        Fields::Named(_) => {}
        Fields::Unnamed(_) => {
            return quote! {
                #item_struct

                const _: () = {
                    compile_error!("aphelion_model only supports structs with named fields");
                };
            };
        }
        Fields::Unit => {
            return quote! {
                #item_struct

                const _: () = {
                    compile_error!("aphelion_model only supports structs with fields");
                };
            };
        }
    }

    let config_field_info = match &item_struct.fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .find(|field| field.ident.as_ref().map(|i| i == "config").unwrap_or(false))
            .map(|f| (f.ident.as_ref().unwrap(), &f.ty)),
        _ => None,
    };

    let (config_field_name, config_field_type) = match config_field_info {
        Some((fname, ftype)) => (fname, ftype),
        None => {
            return quote! {
                #item_struct

                const _: () = {
                    compile_error!(
                        "aphelion_model requires a 'config' field of type ModelConfig. \
                        Expected: config: ModelConfig"
                    );
                };
            };
        }
    };

    // Verify the config field is actually ModelConfig
    if !is_model_config(config_field_type) {
        return quote! {
            #item_struct

            const _: () = {
                compile_error!(
                    "aphelion_model found 'config' field but it is not of type ModelConfig. \
                    Please use: config: aphelion_core::config::ModelConfig"
                );
            };
        };
    }

    // Generate the impl blocks
    let config_field_ref = config_field_name;
    let expanded = quote! {
        #item_struct

        // Implementation of ConfigSpec trait
        // This trait allows the model to provide its configuration.
        #[automatically_derived]
        impl #impl_generics aphelion_core::config::ConfigSpec for #name #ty_generics #where_clause {
            fn config(&self) -> &aphelion_core::config::ModelConfig {
                &self.#config_field_ref
            }
        }

        // Implementation of ModelBuilder trait
        // This trait allows the model to be built into a computation graph.
        // Note: Ensure your struct implements the following method:
        //   fn build_graph(
        //       &self,
        //       backend: &dyn aphelion_core::backend::Backend,
        //       trace: &dyn aphelion_core::diagnostics::TraceSink,
        //   ) -> aphelion_core::graph::BuildGraph
        #[automatically_derived]
        impl #impl_generics aphelion_core::backend::ModelBuilder for #name #ty_generics #where_clause {
            type Output = aphelion_core::graph::BuildGraph;

            fn build(
                &self,
                backend: &dyn aphelion_core::backend::Backend,
                trace: &dyn aphelion_core::diagnostics::TraceSink,
            ) -> Self::Output {
                // This calls your build_graph method. If it's not defined, you'll get a
                // compile error that says "no method named `build_graph` found".
                // Make sure your impl block includes:
                //   fn build_graph(&self, backend: &dyn Backend, trace: &dyn TraceSink) -> BuildGraph { ... }
                self.build_graph(backend, trace)
            }
        }
    };

    expanded
}
