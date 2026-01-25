use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Fields, Item, ItemStruct};

/// Attribute macro for marking a model builder type.
///
/// Generates a `ConfigSpec` impl and a `ModelBuilder` impl
/// that forwards to `build_graph(&self, backend, trace)`.
#[proc_macro_attribute]
pub fn aphelion_model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as Item);
    let output = match input {
        Item::Struct(item_struct) => expand_model_struct(item_struct),
        other => quote! { #other },
    };
    TokenStream::from(output)
}

fn expand_model_struct(item_struct: ItemStruct) -> proc_macro2::TokenStream {
    let name = &item_struct.ident;
    let has_config_field = match &item_struct.fields {
        Fields::Named(fields) => fields
            .named
            .iter()
            .any(|field| field.ident.as_ref().map(|i| i == "config").unwrap_or(false)),
        _ => false,
    };

    if !has_config_field {
        return quote! {
            #item_struct
            compile_error!("aphelion_model requires a `config` field on the struct");
        };
    }

    quote! {
        #item_struct

        impl aphelion_core::config::ConfigSpec for #name {
            fn config(&self) -> &aphelion_core::config::ModelConfig {
                &self.config
            }
        }

        impl aphelion_core::backend::ModelBuilder for #name {
            type Output = aphelion_core::graph::BuildGraph;

            fn build(
                &self,
                backend: &dyn aphelion_core::backend::Backend,
                trace: &dyn aphelion_core::diagnostics::TraceSink,
            ) -> Self::Output {
                self.build_graph(backend, trace)
            }
        }
    }
}
