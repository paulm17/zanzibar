use lazy_static::lazy_static;
use regex::Regex;
use rhai::{Engine, Dynamic, Map, Scope};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::sync::Arc;

const MAX_DEPTH: usize = 100;

lazy_static! {
    static ref DEBUG_ENABLED: bool = env::var("ZANZIBAR_DEBUG")
        .map(|v| v.parse::<bool>().unwrap_or(false))
        .unwrap_or(false);
}

fn debug_print(msg: &str) {
    if *DEBUG_ENABLED {
        println!("[DEBUG] {}", msg);
    }
}

#[derive(Debug, Clone, PartialEq)]
enum AttrValue {
    Int(i32),
    Double(f64),
    Bool(bool),
}

impl std::fmt::Display for AttrValue {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            AttrValue::Int(i) => write!(f, "{}", i),
            AttrValue::Double(d) => write!(f, "{}", d),
            AttrValue::Bool(b) => write!(f, "{}", b),
        }
    }
}

/// Rules are stored with their parameter names (extracted without type annotations)
#[derive(Debug)]
struct Entity {
    #[allow(dead_code)]
    name: String,
    relations: HashMap<String, String>,
    permissions: HashMap<String, String>,
    // rule name -> (parameter names, rule body)
    rules: HashMap<String, (Vec<String>, String)>,
}

fn parse_schema(schema_str: &str) -> HashMap<String, Entity> {
    let rule_re = Regex::new(r"rule\s+(\w+)\s*\(([^)]*)\)\s*\{([^}]*)\}").unwrap();
    let mut global_rules: HashMap<String, (Vec<String>, String)> = HashMap::new();
    for cap in rule_re.captures_iter(schema_str) {
        let rule_name = cap[1].to_string();
        let params_str = cap[2].trim();
        let body = cap[3].trim().to_string();
        let params: Vec<String> = if params_str.is_empty() {
            Vec::new()
        } else {
            params_str.split(',')
                .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
                .collect()
        };
        global_rules.insert(rule_name, (params, body));
    }

    let mut entities = HashMap::new();
    let entity_re = Regex::new(r"entity\s+(\w+)\s*\{([^}]*)\}").unwrap();
    for cap in entity_re.captures_iter(schema_str) {
        let entity_name = cap[1].to_string();
        let block_content = cap[2].trim();
        let mut relations = HashMap::new();
        let mut permissions = HashMap::new();
        let mut rules = HashMap::new();
        for line in block_content.lines() {
            let line = line.trim();
            if line.starts_with("//") || line.is_empty() {
                continue;
            }
            if line.starts_with("relation") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    let rel_name = parts[1];
                    let targets: Vec<String> = parts
                        .iter()
                        .filter(|&&s| s.starts_with('@'))
                        .map(|s| s.to_string())
                        .collect();
                    relations.insert(rel_name.to_string(), targets.join(" "));
                }
            } else if line.starts_with("permission") || line.starts_with("action") {
                let parts: Vec<&str> = line.split('=').collect();
                if parts.len() >= 2 {
                    let left = parts[0].trim();
                    let tokens: Vec<&str> = left.split_whitespace().collect();
                    if tokens.len() >= 2 {
                        let perm_name = tokens[1];
                        let expr = parts[1].trim();
                        permissions.insert(perm_name.to_string(), expr.to_string());
                    }
                }
            } else if line.starts_with("rule") {
                if let Some(cap) = rule_re.captures(line) {
                    let rule_name = cap[1].to_string();
                    let params_str = cap[2].trim();
                    let body = cap[3].trim().to_string();
                    let params: Vec<String> = if params_str.is_empty() {
                        Vec::new()
                    } else {
                        params_str.split(',')
                            .map(|s| s.trim().split_whitespace().next().unwrap_or("").to_string())
                            .collect()
                    };
                    rules.insert(rule_name, (params, body));
                }
            }
        }
        let entity = Entity {
            name: entity_name.clone(),
            relations,
            permissions,
            rules,
        };
        entities.insert(entity_name, entity);
    }
    if !global_rules.is_empty() {
        entities.insert(
            "__global__".to_string(),
            Entity {
                name: "__global__".to_string(),
                relations: HashMap::new(),
                permissions: HashMap::new(),
                rules: global_rules,
            },
        );
    }
    entities
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Subject {
    entity: String,
    id: String,
}

impl Subject {
    fn from_str(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.trim().split(':').collect();
        if parts.len() != 2 {
            None
        } else {
            Some(Subject {
                entity: parts[0].to_string(),
                id: parts[1].to_string(),
            })
        }
    }
}

struct RelationshipStore {
    store: HashMap<(String, String, String), Vec<Subject>>,
}

impl RelationshipStore {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
    fn insert(&mut self, entity: &str, id: &str, relation: &str, subject: Subject) {
        let key = (entity.to_string(), id.to_string(), relation.to_string());
        self.store.entry(key).or_default().push(subject);
    }
    fn get(&self, entity: &str, id: &str, relation: &str) -> Vec<Subject> {
        self.store
            .get(&(entity.to_string(), id.to_string(), relation.to_string()))
            .cloned()
            .unwrap_or_default()
    }
}

struct AttributeStore {
    store: HashMap<(String, String), HashMap<String, AttrValue>>,
}

impl AttributeStore {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
    fn set_int(&mut self, entity: &str, id: &str, attr: &str, value: i32) {
        self.store
            .entry((entity.to_string(), id.to_string()))
            .or_default()
            .insert(attr.to_string(), AttrValue::Int(value));
    }
    fn set_double(&mut self, entity: &str, id: &str, attr: &str, value: f64) {
        self.store
            .entry((entity.to_string(), id.to_string()))
            .or_default()
            .insert(attr.to_string(), AttrValue::Double(value));
    }
    fn set_bool(&mut self, entity: &str, id: &str, attr: &str, value: bool) {
        self.store
            .entry((entity.to_string(), id.to_string()))
            .or_default()
            .insert(attr.to_string(), AttrValue::Bool(value));
    }
    fn get(&self, entity: &str, id: &str, attr: &str) -> Option<&AttrValue> {
        self.store
            .get(&(entity.to_string(), id.to_string()))
            .and_then(|attrs| attrs.get(attr))
    }
}

fn load_relationships_and_attributes(
    relationships_str: &str,
    attr_store: &mut AttributeStore,
) -> RelationshipStore {
    let mut store = RelationshipStore::new();
    for line in relationships_str.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        if line.contains('#') {
            let mut parts = line.splitn(2, '#');
            let left = parts.next().unwrap();
            let right = match parts.next() {
                Some(r) => r,
                None => continue,
            };
            let mut left_parts = left.splitn(2, ':');
            let entity = left_parts.next().unwrap();
            let id = match left_parts.next() {
                Some(id) => id,
                None => continue,
            };
            let mut right_parts = right.splitn(2, '@');
            let relation = right_parts.next().unwrap().trim();
            let subject_str = match right_parts.next() {
                Some(s) => s.trim(),
                None => continue,
            };
            if let Some(subject) = Subject::from_str(subject_str) {
                store.insert(entity, id, relation, subject);
            }
        } else if line.contains('$') {
            let mut parts = line.splitn(2, '$');
            let left = parts.next().unwrap();
            let right = match parts.next() {
                Some(r) => r,
                None => continue,
            };
            let mut left_parts = left.splitn(2, ':');
            let entity = left_parts.next().unwrap();
            let id = match left_parts.next() {
                Some(id) => id,
                None => continue,
            };
            let mut attr_parts = right.splitn(2, '|');
            let attr_name = attr_parts.next().unwrap().trim();
            let rest = match attr_parts.next() {
                Some(r) => r.trim(),
                None => continue,
            };
            let mut type_parts = rest.splitn(2, ':');
            let attr_type = type_parts.next().unwrap().trim();
            let attr_value = match type_parts.next() {
                Some(v) => v.trim(),
                None => continue,
            };
            if attr_type.eq_ignore_ascii_case("boolean") {
                let value = attr_value.eq_ignore_ascii_case("true");
                attr_store.set_bool(entity, id, attr_name, value);
            } else if attr_type.eq_ignore_ascii_case("integer") {
                if let Ok(val) = attr_value.parse::<i32>() {
                    attr_store.set_int(entity, id, attr_name, val);
                }
            } else if attr_type.eq_ignore_ascii_case("double") {
                if let Ok(val) = attr_value.parse::<f64>() {
                    attr_store.set_double(entity, id, attr_name, val);
                }
            }
        }
    }
    store
}

fn get_attr_as_double(
    attr_store: &AttributeStore,
    entity: &str,
    id: &str,
    attr: &str,
) -> f64 {
    if let Some(val) = attr_store.get(entity, id, attr) {
        match val {
            AttrValue::Int(i) => *i as f64,
            AttrValue::Double(d) => *d,
            AttrValue::Bool(b) => {
                if *b {
                    1.0
                } else {
                    0.0
                }
            }
        }
    } else {
        0.0
    }
}

fn get_relation_subjects(
    rel_store: &RelationshipStore,
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    relation: &str,
) -> Vec<Subject> {
    let direct_subjects = rel_store.get(entity_name, instance_id, relation);
    if !direct_subjects.is_empty() {
        if let Some(entity_def) = schema.get(entity_name) {
            if entity_def
                .relations
                .get(relation)
                .map_or(false, |t| t.contains('#'))
            {
                let mut subjects = Vec::new();
                for subject in direct_subjects {
                    if subject.id.contains('#') {
                        let parts: Vec<&str> = subject.id.splitn(2, '#').collect();
                        let base_id = parts[0];
                        let chained_relation = parts[1];
                        let resolved_subjects = get_relation_subjects(
                            rel_store,
                            schema,
                            &subject.entity,
                            base_id,
                            chained_relation,
                        );
                        debug_print(&format!(
                            "Resolved chained subject {:?} -> {:?}",
                            subject, resolved_subjects
                        ));
                        subjects.extend(resolved_subjects);
                    } else {
                        subjects.push(subject);
                    }
                }
                return subjects;
            }
        }
        return direct_subjects;
    } else {
        if let Some(entity_def) = schema.get(entity_name) {
            if let Some(rel_def) = entity_def.relations.get(relation) {
                let tokens: Vec<&str> = rel_def.split_whitespace().collect();
                if tokens.len() == 1 && tokens[0].starts_with('@') {
                    let target_type = tokens[0].trim_start_matches('@');
                    let mut results = Vec::new();
                    for ((src_entity, src_id, _), subjects) in &rel_store.store {
                        if src_entity == target_type {
                            for subj in subjects {
                                if subj.entity == entity_name && subj.id == instance_id {
                                    results.push(Subject {
                                        entity: src_entity.clone(),
                                        id: src_id.clone(),
                                    });
                                }
                            }
                        }
                    }
                    if !results.is_empty() {
                        debug_print(&format!(
                            "Reverse lookup for {}:{} via '{}' found {:?}",
                            entity_name, instance_id, relation, results
                        ));
                    }
                    return results;
                }
            }
        }
    }
    direct_subjects
}

fn split_top_level<'a>(expr: &'a str, delimiter: &str) -> Vec<&'a str> {
    let mut tokens = Vec::new();
    let mut start = 0;
    let mut depth = 0;
    let chars: Vec<char> = expr.chars().collect();
    let delim_chars: Vec<char> = delimiter.chars().collect();
    let delim_len = delim_chars.len();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c == '(' {
            depth += 1;
        } else if c == ')' {
            if depth > 0 {
                depth -= 1;
            }
        }
        if depth == 0 && i + delim_len <= chars.len() {
            let candidate: Vec<char> = chars[i..i + delim_len].to_vec();
            if candidate == delim_chars {
                tokens.push(expr[start..i].trim());
                i += delim_len;
                start = i;
                continue;
            }
        }
        i += 1;
    }
    tokens.push(expr[start..].trim());
    tokens
}

fn ensure_balanced(expr: &str) -> String {
    let mut count = 0;
    for c in expr.chars() {
        if c == '(' {
            count += 1;
        } else if c == ')' {
            if count > 0 {
                count -= 1;
            }
        }
    }
    let mut result = expr.trim().to_string();
    for _ in 0..count {
        result.push(')');
    }
    result
}

fn remove_outer_parens(expr: &str) -> String {
    let expr = ensure_balanced(expr);
    let chars: Vec<char> = expr.chars().collect();
    if chars.len() >= 2 && chars[0] == '(' && chars[chars.len() - 1] == ')' {
        let mut count = 0;
        for (i, &c) in chars.iter().enumerate() {
            if c == '(' {
                count += 1;
            } else if c == ')' {
                count -= 1;
            }
            if count == 0 {
                if i == chars.len() - 1 {
                    let inner = &expr[1..expr.len() - 1];
                    return inner.trim().to_string();
                } else {
                    break;
                }
            }
        }
    }
    expr
}

fn evaluate_expr(
    expr: &str,
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
    context: &Map,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        debug_print("Max recursion depth reached in evaluate_expr");
        return false;
    }
    debug_print(&format!("Evaluating expr: '{}'", expr));
    let expr = remove_outer_parens(expr);
    let or_tokens = split_top_level(&expr, " or ");
    debug_print(&format!("OR tokens: {:?}", or_tokens));
    if or_tokens.len() > 1 {
        for token in or_tokens {
            if evaluate_expr(token, schema, entity_name, instance_id, user, rel_store, attr_store, context, depth + 1) {
                return true;
            }
        }
        return false;
    }
    let and_tokens = split_top_level(&expr, " and ");
    debug_print(&format!("AND tokens: {:?}", and_tokens));
    if and_tokens.len() > 1 {
        for token in and_tokens {
            if !evaluate_expr(token, schema, entity_name, instance_id, user, rel_store, attr_store, context, depth + 1) {
                return false;
            }
        }
        return true;
    }
    evaluate_token(&expr, schema, entity_name, instance_id, user, rel_store, attr_store, context, depth + 1)
}

fn evaluate_token(
    token: &str,
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
    context: &Map,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        debug_print("Max recursion depth reached in evaluate_token");
        return false;
    }
    let token = token.trim();

    if token.contains('(') && token.ends_with(')') {
        let open_paren = token.find('(').unwrap();
        let func_name = token[..open_paren].trim();
        let args_str = &token[open_paren + 1..token.len() - 1];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
        return evaluate_function(
            func_name,
            &args,
            schema,
            entity_name,
            instance_id,
            attr_store,
            context,
            user,
            rel_store,
            depth + 1,
        );
    }

    if token.starts_with("not ") {
        let subtoken = token[4..].trim();
        return !evaluate_expr(
            subtoken,
            schema,
            entity_name,
            instance_id,
            user,
            rel_store,
            attr_store,
            context,
            depth + 1,
        );
    }

    if token.contains(" not ") {
        let parts: Vec<&str> = token.splitn(2, " not ").collect();
        let left = parts[0].trim();
        let right = parts[1].trim();
        return evaluate_expr(
            left,
            schema,
            entity_name,
            instance_id,
            user,
            rel_store,
            attr_store,
            context,
            depth + 1,
        ) && !evaluate_expr(
            right,
            schema,
            entity_name,
            instance_id,
            user,
            rel_store,
            attr_store,
            context,
            depth + 1,
        );
    }

    if token.contains('.') {
        let parts: Vec<&str> = token.splitn(2, '.').collect();
        if parts.len() == 2 {
            let rel_name = parts[0].trim();
            let nested_perm = parts[1].trim();
            debug_print(&format!(
                "Token '{}' -> relation '{}' and nested permission '{}'",
                token, rel_name, nested_perm
            ));
            let subjects =
                get_relation_subjects(rel_store, schema, entity_name, instance_id, rel_name);
            debug_print(&format!(
                "Subjects for {}:{} via '{}' -> {:?}",
                entity_name, instance_id, rel_name, subjects
            ));
            for subj in subjects {
                let res = evaluate_permission(
                    schema,
                    &subj.entity,
                    &subj.id,
                    nested_perm,
                    user,
                    rel_store,
                    attr_store,
                    context,
                    depth + 1,
                );
                debug_print(&format!(
                    "Evaluating nested permission {} on {}:{} for {} -> {}",
                    nested_perm, subj.entity, subj.id, user, res
                ));
                if res {
                    return true;
                }
            }
            return false;
        }
    }
    let subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, token);
    debug_print(&format!(
        "Subjects for {}:{} via '{}' -> {:?}",
        entity_name, instance_id, token, subjects
    ));
    if !subjects.is_empty() {
        if subjects.iter().any(|s| format!("{}:{}", s.entity, s.id) == user) {
            debug_print(&format!("User {} found in subjects for token '{}'", user, token));
            return true;
        }
        return false;
    }
    if let Some(val) = attr_store.get(entity_name, instance_id, token) {
        debug_print(&format!(
            "Found attribute {} on {}:{} with value {:?}",
            token, entity_name, instance_id, val
        ));
        return match val {
            AttrValue::Int(i) => *i != 0,
            AttrValue::Double(d) => *d != 0.0,
            AttrValue::Bool(b) => *b,
        };
    }
    false
}

fn evaluate_function(
    func_name: &str,
    args: &[&str],
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    attr_store: &AttributeStore,
    context: &Map,
    user: &str,
    rel_store: &RelationshipStore,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        debug_print("Max recursion depth reached in evaluate_function");
        return false;
    }
    // New built-in function for tuple-to-userset rewriting.
    if func_name == "tuple_to_userset" {
        if args.len() == 2 {
            return evaluate_tuple_to_userset(
                args[0],
                args[1],
                schema,
                entity_name,
                instance_id,
                user,
                rel_store,
                attr_store,
                context,
                depth + 1,
            );
        } else {
            debug_print("tuple_to_userset expects 2 arguments");
            return false;
        }
    }
    if let Some((param_names, rule_body)) = get_rule_body(schema, func_name) {
        debug_print(&format!("Evaluating rule {} with body: {}", func_name, rule_body));
        if param_names.len() != args.len() {
            debug_print(&format!(
                "Argument count mismatch for {}: expected {} but got {}",
                func_name,
                param_names.len(),
                args.len()
            ));
            return false;
        }
        let engine = Engine::new();
        let mut scope = Scope::new();
        for (i, param) in param_names.iter().enumerate() {
            let arg = args[i];
            let val = get_attr_as_double(attr_store, entity_name, instance_id, arg);
            scope.push_dynamic(param.to_string(), Dynamic::from_float(val));
        }
        scope.push("context", Dynamic::from(context.clone()));
        match engine.eval_with_scope::<bool>(&mut scope, &rule_body) {
            Ok(result) => result,
            Err(err) => {
                debug_print(&format!("Error evaluating rule {}: {}", func_name, err));
                false
            }
        }
    } else {
        debug_print(&format!("Rule {} not found", func_name));
        false
    }
}

fn evaluate_tuple_to_userset(
    tupleset_relation: &str,
    computed_relation: &str,
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
    context: &Map,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        debug_print("Max recursion depth reached in evaluate_tuple_to_userset");
        return false;
    }
    let subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, tupleset_relation);
    debug_print(&format!(
        "tuple_to_userset: Found subjects for tupleset '{}': {:?}",
        tupleset_relation, subjects
    ));
    for subj in subjects {
        let res = evaluate_permission(
            schema,
            &subj.entity,
            &subj.id,
            computed_relation,
            user,
            rel_store,
            attr_store,
            context,
            depth + 1,
        );
        debug_print(&format!(
            "tuple_to_userset: Evaluating computed_relation '{}' on {}:{} for {} -> {}",
            computed_relation, subj.entity, subj.id, user, res
        ));
        if res {
            return true;
        }
    }
    false
}

fn get_rule_body(schema: &HashMap<String, Entity>, rule_name: &str) -> Option<(Vec<String>, String)> {
    for entity in schema.values() {
        if let Some((params, body)) = entity.rules.get(rule_name) {
            return Some((params.clone(), body.clone()));
        }
    }
    None
}

fn evaluate_permission(
    schema: &HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    perm_name: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
    context: &Map,
    depth: usize,
) -> bool {
    if depth > MAX_DEPTH {
        debug_print("Max recursion depth reached in evaluate_permission");
        return false;
    }
    if let Some(entity) = schema.get(entity_name) {
        if let Some(expr) = entity.permissions.get(perm_name) {
            debug_print(&format!(
                "Evaluating permission {} on {}:{} with expression '{}'",
                perm_name, entity_name, instance_id, expr
            ));
            return evaluate_expr(expr, schema, entity_name, instance_id, user, rel_store, attr_store, context, depth + 1);
        }
    }
    let direct_subjects = rel_store.get(entity_name, instance_id, perm_name);
    let resolved_subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, perm_name);
    direct_subjects
        .into_iter()
        .chain(resolved_subjects.into_iter())
        .any(|s| format!("{}:{}", s.entity, s.id) == user)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;
    use std::error::Error;
    use std::sync::Arc;

    #[derive(Debug, Deserialize)]
    struct TestFile {
        schema: String,
        relationships: Vec<String>,
        attributes: Vec<serde_yaml::Value>,
        scenarios: Vec<Scenario>,
    }

    #[derive(Debug, Deserialize)]
    struct Scenario {
        name: String,
        checks: Vec<Check>,
        #[serde(default)]
        entity_filters: Vec<String>,
        #[serde(default)]
        subject_filters: Vec<String>,
    }

    #[derive(Debug, Deserialize)]
    struct Check {
        entity: String,
        subject: String,
        #[serde(default)]
        context: Option<serde_yaml::Value>,
        assertions: HashMap<String, Option<bool>>,
    }

    fn yaml_value_to_dynamic(value: &serde_yaml::Value) -> Dynamic {
        match value {
            serde_yaml::Value::Null => Dynamic::UNIT,
            serde_yaml::Value::Bool(b) => Dynamic::from_bool(*b),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Dynamic::from_float(i as f64)
                } else if let Some(f) = n.as_f64() {
                    Dynamic::from_float(f)
                } else {
                    Dynamic::UNIT
                }
            }
            serde_yaml::Value::String(s) => Dynamic::from(s.clone()),
            serde_yaml::Value::Sequence(seq) => {
                let vec: Vec<Dynamic> = seq.iter().map(yaml_value_to_dynamic).collect();
                Dynamic::from_array(vec)
            }
            serde_yaml::Value::Mapping(map) => {
                let mut m = Map::new();
                for (k, v) in map {
                    if let serde_yaml::Value::String(key_str) = k {
                        m.insert(key_str.into(), yaml_value_to_dynamic(v));
                    }
                }
                Dynamic::from(m)
            }
            _ => Dynamic::UNIT,
        }
    }
    
    fn yaml_value_to_rhai_map(value: &serde_yaml::Value) -> Map {
        if let serde_yaml::Value::Mapping(map) = value {
            let mut rhai_map = Map::new();
            for (k, v) in map {
                if let serde_yaml::Value::String(key_str) = k {
                    rhai_map.insert(key_str.into(), yaml_value_to_dynamic(v));
                }
            }
            rhai_map
        } else {
            Map::new()
        }
    }

    fn create_engine_from_yaml(test_file: &TestFile) -> (Engine, Arc<HashMap<String, Entity>>, Arc<RelationshipStore>, Arc<AttributeStore>) {
        let relationships_str = test_file.relationships.join("\n");
        let attributes_str = test_file
            .attributes
            .iter()
            .map(|attr| {
                if let Some(s) = attr.as_str() {
                    s.to_string()
                } else {
                    serde_yaml::to_string(attr)
                        .unwrap_or_else(|_| String::new())
                        .trim_matches('"')
                        .to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let combined = if attributes_str.is_empty() {
            relationships_str
        } else {
            format!("{}\n{}", relationships_str, attributes_str)
        };
        create_engine::<fn(&mut AttributeStore)>(&test_file.schema, &combined, None)
    }

    fn create_engine<F>(
        schema_str: &str,
        relationships_str: &str,
        attr_setup: Option<F>,
    ) -> (Engine, Arc<HashMap<String, Entity>>, Arc<RelationshipStore>, Arc<AttributeStore>)
    where
        F: FnOnce(&mut AttributeStore),
    {
        let schema = Arc::new(parse_schema(schema_str));
        let mut attr_store = AttributeStore::new();
        let rel_store = Arc::new(load_relationships_and_attributes(relationships_str, &mut attr_store));
        if let Some(setup) = attr_setup {
            setup(&mut attr_store);
        }
        let attr_store = Arc::new(attr_store);
        let mut engine = Engine::new();
        let schema_for_rhai = Arc::clone(&schema);
        let rel_store_rhai = Arc::clone(&rel_store);
        let attr_store_rhai = Arc::clone(&attr_store);

        // Register the check_permission function that accepts a context map.
        {
            let schema_clone = Arc::clone(&schema_for_rhai);
            let rel_store_clone = Arc::clone(&rel_store_rhai);
            let attr_store_clone = Arc::clone(&attr_store_rhai);
            engine.register_fn(
                "check_permission",
                move |entity: &str, id: &str, perm: &str, user: &str, context: Map| -> bool {
                    evaluate_permission(
                        &schema_clone,
                        entity,
                        id,
                        perm,
                        user,
                        &rel_store_clone,
                        &attr_store_clone,
                        &context,
                        0,
                    )
                },
            );
        }
        (engine, schema, rel_store, attr_store)
    }

    #[test]
    fn test_zanzibar_permissions() -> Result<(), Box<dyn Error>> {
        println!("Running Zanzibar permission tests:\n");

        // Extended list of shape files, including new ones for tuple rewriting.
        let shape_files = vec![
            "banking",
            "disney_plus",
            "mercury",
            "organisation",
            "rbac",
            "user_groups",
            "custom_roles",
            "google_docs",
            "instagram",
            "document_management",
            "hospital_management",
            "content_moderation",
            "document_hierarchy",
        ];

        let mut results = Vec::new();

        for file_stem in shape_files {
            let path = format!("shapes/{}.yaml", file_stem);
            let test_data = fs::read_to_string(&path)?;
            let test_file: TestFile = serde_yaml::from_str(&test_data)?;

            let (engine, _schema, _rel_store, _attr_store) = create_engine_from_yaml(&test_file);

            let mut errors = Vec::new();

            for scenario in test_file.scenarios {
                for check in scenario.checks {
                    if !scenario.entity_filters.is_empty()
                        && !scenario.entity_filters.iter().any(|f| check.entity.contains(f))
                    {
                        continue;
                    }
                    if !scenario.subject_filters.is_empty()
                        && !scenario.subject_filters.iter().any(|f| check.subject.contains(f))
                    {
                        continue;
                    }

                    let parts: Vec<&str> = check.entity.splitn(2, ':').collect();
                    if parts.len() != 2 {
                        errors.push(format!("Invalid entity format: {}", check.entity));
                        continue;
                    }
                    let entity_type = parts[0];
                    let entity_id = parts[1];
                    let user = &check.subject;

                    let context_value = check.context.clone().unwrap_or(serde_yaml::Value::Mapping(Default::default()));
                    let context_map = yaml_value_to_rhai_map(&context_value);

                    for (perm, expected_opt) in check.assertions {
                        if let Some(expected) = expected_opt {
                            let mut scope = Scope::new();
                            scope.push("context", context_map.clone());
                            let script = format!(
                                "check_permission(\"{}\", \"{}\", \"{}\", \"{}\", context)",
                                entity_type, entity_id, perm, user
                            );
                            let res = engine.eval_with_scope::<bool>(&mut scope, &script).unwrap_or(false);
                            if res != expected {
                                errors.push(format!(
                                    "Scenario '{}': Permission '{}' on {}:{} for {} => expected: {}, got: {}",
                                    scenario.name, perm, entity_type, entity_id, user, expected, res
                                ));
                            }
                        }
                    }
                }
            }

            let passed = errors.is_empty();
            println!("{:<20} {}", file_stem, if passed { "✅ PASSED" } else { "❌ FAILED" });
            if !passed {
                for error in &errors {
                    println!("  └─ {}", error);
                }
            }
            results.push((file_stem.to_string(), passed, errors));
        }

        let (passed, failed): (Vec<_>, Vec<_>) =
            results.iter().partition(|(_, p, _)| *p);

        println!("\nTest Summary:");
        println!("  Passed: {}", passed.len());
        println!("  Failed: {}", failed.len());
        println!("  Total:  {}", passed.len() + failed.len());

        if !failed.is_empty() {
            panic!("{} tests failed", failed.len());
        }

        Ok(())
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let schema_dsl = fs::read_to_string("user_groups_schema.yaml")?;
    let schema = parse_schema(&schema_dsl);
    println!("Parsed schema entities:");
    for (name, entity) in &schema {
        println!("Entity: {} => {:?}", name, entity);
    }

    let relationships_str = fs::read_to_string("user_groups_relationships.yaml")?;
    let mut attr_store = AttributeStore::new();
    let rel_store = Arc::new(load_relationships_and_attributes(&relationships_str, &mut attr_store));

    attr_store.set_int("state", "1", "age_limit", 18);
    attr_store.set_int("patient", "1", "age", 15);

    let schema_for_rhai = Arc::new(schema);
    let rel_store_rhai = Arc::clone(&rel_store);
    let attr_store_rhai = Arc::new(attr_store);

    let schema_clone = Arc::clone(&schema_for_rhai);
    let rel_store_clone = Arc::clone(&rel_store_rhai);
    let attr_store_clone = Arc::clone(&attr_store_rhai);
    let mut engine = Engine::new();
    engine.register_fn(
        "check_permission",
        move |entity: &str, id: &str, perm: &str, user: &str, context: Map| -> bool {
            evaluate_permission(
                &schema_clone,
                entity,
                id,
                perm,
                user,
                &rel_store_clone,
                &attr_store_clone,
                &context,
                100
            )
        },
    );

    let mut context = Map::new();
    context.insert("amount".into(), Dynamic::from_float(3000.0));

    let test_result = evaluate_permission(
        &schema_for_rhai,
        "account",
        "1",
        "withdraw",
        "user:andrew",
        &rel_store_rhai,
        &attr_store_rhai,
        &context,
        100
    );
    println!("Permission check result for user:andrew: {}", test_result);

    Ok(())
}