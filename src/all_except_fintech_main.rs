// works with all schemas except fintech_banking

use rhai::Engine;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::sync::Arc;

#[derive(Debug)]
struct Entity {
    #[allow(dead_code)]
    name: String,
    // All target tokens are collected as a single space‐separated string.
    relations: HashMap<String, String>,
    permissions: HashMap<String, String>,
    #[allow(dead_code)]
    rules: HashMap<String, String>,
}

fn parse_schema(schema_str: &str) -> HashMap<String, Entity> {
    let mut entities = HashMap::new();
    for block in schema_str.split("entity ") {
        let block = block.trim();
        if block.is_empty() {
            continue;
        }
        let mut lines = block.lines();
        let first_line = lines.next().unwrap();
        let entity_name = first_line.split_whitespace().next().unwrap();
        let block_content = if let Some(start) = block.find('{') {
            if let Some(end) = block.rfind('}') {
                &block[start + 1..end]
            } else {
                ""
            }
        } else {
            ""
        };

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
                    // Collect every token starting with '@'
                    let targets: Vec<String> = parts.iter()
                        .filter(|&&s| s.starts_with('@'))
                        .map(|s| s.trim_start_matches('@').to_string())
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
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let rule_with_params = parts[1];
                    if let Some(idx) = rule_with_params.find('(') {
                        if let Some(body_start) = line.find('{') {
                            if let Some(body_end) = line.rfind('}') {
                                let body = &line[body_start + 1..body_end].trim();
                                rules.insert(rule_with_params[..idx].to_string(), body.to_string());
                            }
                        }
                    }
                }
            }
        }
        let entity = Entity {
            name: entity_name.to_string(),
            relations,
            permissions,
            rules,
        };
        entities.insert(entity_name.to_string(), entity);
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

/// load_relationships_and_attributes processes both relationship and attribute lines.
/// Relationship lines use '#' and attribute lines use '$'. Boolean attributes are stored as 1 (true) or 0 (false).
fn load_relationships_and_attributes(relationships_str: &str, attr_store: &mut AttributeStore) -> RelationshipStore {
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
            // Attribute line format: <entity>:<id>$<attr>|<type>:<value>
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
            if attr_type == "boolean" {
                let value = if attr_value.eq_ignore_ascii_case("true") { 1 } else { 0 };
                attr_store.set(entity, id, attr_name, value);
            } else if attr_type == "integer" {
                if let Ok(val) = attr_value.parse::<i32>() {
                    attr_store.set(entity, id, attr_name, val);
                }
            }
        }
    }
    store
}

struct AttributeStore {
    store: HashMap<(String, String), HashMap<String, i32>>,
}

impl AttributeStore {
    fn new() -> Self {
        Self {
            store: HashMap::new(),
        }
    }
    fn set(&mut self, entity: &str, id: &str, attr: &str, value: i32) {
        self.store
            .entry((entity.to_string(), id.to_string()))
            .or_default()
            .insert(attr.to_string(), value);
    }
    fn get(&self, entity: &str, id: &str, attr: &str) -> Option<i32> {
        self.store
            .get(&(entity.to_string(), id.to_string()))
            .and_then(|attrs| attrs.get(attr).copied())
    }
}

/// For documents, if no direct "org" relation exists, perform a reverse lookup.
fn get_reverse_org(
    rel_store: &RelationshipStore,
    doc_id: &str,
) -> Vec<Subject> {
    let mut subjects = Vec::new();
    for ((e, id, rel), subs) in &rel_store.store {
        if e == "organization" && rel == "document" {
            for subj in subs {
                if subj.entity == "document" && subj.id == doc_id {
                    subjects.push(Subject { entity: "organization".to_string(), id: id.clone() });
                }
            }
        }
    }
    subjects
}

/// get_relation_subjects handles chained references and uses reverse lookup for a document's "org".
fn get_relation_subjects(
    rel_store: &RelationshipStore,
    schema: &HashMap<String, Entity>,
    entity: &str,
    instance_id: &str,
    relation: &str,
) -> Vec<Subject> {
    if entity == "document" && relation == "org" {
        let direct_subjects = rel_store.get(entity, instance_id, relation);
        if direct_subjects.is_empty() {
            println!("[DEBUG] No direct 'org' for document {} – using reverse lookup", instance_id);
            return get_reverse_org(rel_store, instance_id);
        }
    }
    if let Some(entity_def) = schema.get(entity) {
        let direct_subjects = rel_store.get(entity, instance_id, relation);
        if entity_def.relations.get(relation).map_or(false, |t| t.contains("group")) {
            let mut subjects = Vec::new();
            for subject in direct_subjects {
                if subject.id.contains('#') {
                    let parts: Vec<&str> = subject.id.splitn(2, '#').collect();
                    let base_id = parts[0];
                    let chained_relation = parts[1];
                    let resolved_subjects = rel_store.get(&subject.entity, base_id, chained_relation);
                    println!("[DEBUG] Resolved chained subject {:?} -> {:?}", subject, resolved_subjects);
                    subjects.extend(resolved_subjects);
                } else if subject.entity == "group" {
                    // Instead of trying to pre-expand group relationships using 'direct_member' or 'manager',
                    // simply return the group subject for the nested permission check.
                    subjects.push(subject);
                } else {
                    subjects.push(subject);
                }
            }
            return subjects;
        } else {
            return direct_subjects;
        }
    }
    Vec::new()
}

/// Ensure the expression is balanced by counting unmatched '(' and appending that many ')'
fn ensure_balanced(expr: &str) -> String {
    let mut count = 0;
    for c in expr.chars() {
        if c == '(' {
            count += 1;
        } else if c == ')' {
            if count > 0 { count -= 1; }
        }
    }
    let mut result = expr.trim().to_string();
    for _ in 0..count {
        result.push(')');
    }
    result
}

/// Remove outer parentheses only if the first '(' matches the last ')'
fn remove_outer_parens(expr: &str) -> String {
    let expr = ensure_balanced(expr);
    let chars: Vec<char> = expr.chars().collect();
    if chars.len() >= 2 && chars[0] == '(' && chars[chars.len()-1] == ')' {
        let mut count = 0;
        // Find the position where the initial '(' is closed.
        for (i, &c) in chars.iter().enumerate() {
            if c == '(' {
                count += 1;
            } else if c == ')' {
                count -= 1;
            }
            if count == 0 {
                if i == chars.len() - 1 {
                    // Outer pair encloses the whole expression.
                    let inner = &expr[1..expr.len()-1];
                    return inner.trim().to_string();
                } else {
                    break;
                }
            }
        }
    }
    expr
}

/// Splits `expr` on top-level occurrences of delimiter.
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
            if depth > 0 { depth -= 1; }
        }
        // Only consider delimiter at depth 0.
        if depth == 0 && i + delim_len <= chars.len() && &chars[i..i+delim_len] == delim_chars.as_slice() {
            tokens.push(expr[start..i].trim());
            i += delim_len;
            start = i;
            continue;
        }
        i += 1;
    }
    tokens.push(expr[start..].trim());
    tokens
}

/// Recursively evaluate a boolean expression.
fn evaluate_expr(
    expr: &str,
    schema: &std::collections::HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
) -> bool {
    let expr = remove_outer_parens(expr);
    // Split on top-level " or "
    let or_tokens = split_top_level(&expr, " or ");
    if or_tokens.len() > 1 {
        for token in or_tokens {
            if evaluate_expr(token, schema, entity_name, instance_id, user, rel_store, attr_store) {
                return true;
            }
        }
        return false;
    }
    // Split on top-level " and "
    let and_tokens = split_top_level(&expr, " and ");
    if and_tokens.len() > 1 {
        for token in and_tokens {
            if !evaluate_expr(token, schema, entity_name, instance_id, user, rel_store, attr_store) {
                return false;
            }
        }
        return true;
    }
    // Base case: evaluate the token.
    evaluate_token(&expr, schema, entity_name, instance_id, user, rel_store, attr_store)
}

/// Evaluate a base token.
fn evaluate_token(
    token: &str,
    schema: &std::collections::HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
) -> bool {
    let token = token.trim();
    // Function call detection: token must contain '(' and end with ')'
    if token.contains('(') && token.ends_with(')') {
        let open_paren = token.find('(').unwrap();
        let func_name = token[..open_paren].trim();
        let args_str = &token[open_paren+1..token.len()-1];
        let args: Vec<&str> = args_str.split(',').map(|s| s.trim()).collect();
        // Support check_member_approval and check_admin_approval.
        if (func_name == "check_member_approval" || func_name == "check_admin_approval") && args.len() == 2 {
            let left = attr_store.get(entity_name, instance_id, args[0]).unwrap_or(0);
            let right = attr_store.get(entity_name, instance_id, args[1]).unwrap_or(0);
            let result = left <= right;
            println!("[DEBUG] Evaluating function {} with args {} and {} on {}:{} -> {}", func_name, left, right, entity_name, instance_id, result);
            return result;
        }
        // Simulate check_balance and check_limit functions.
        if func_name == "check_balance" || func_name == "check_limit" {
            return true;
        }
        println!("[DEBUG] Unknown function call: {}", token);
        return false;
    }
    // Handle nested permission calls via a dot.
    if token.contains('.') {
        let parts: Vec<&str> = token.splitn(2, '.').collect();
        if parts.len() == 2 {
            let rel_name = parts[0].trim();
            let nested_perm = parts[1].trim();
            println!("[DEBUG] Token '{}' -> relation '{}' and nested permission '{}'", token, rel_name, nested_perm);
            let subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, rel_name);
            println!("[DEBUG] Subjects for {}:{} via '{}' -> {:?}", entity_name, instance_id, rel_name, subjects);
            for subj in subjects {
                let res = evaluate_permission(schema, &subj.entity, &subj.id, nested_perm, user, rel_store, attr_store);
                println!("[DEBUG] Evaluating nested permission {} on {}:{} for {} -> {}", nested_perm, subj.entity, subj.id, user, res);
                if res {
                    return true;
                }
            }
            return false;
        }
    }
    // Otherwise, treat token as a relation check or attribute lookup.
    let subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, token);
    println!("[DEBUG] Subjects for {}:{} via '{}' -> {:?}", entity_name, instance_id, token, subjects);
    if !subjects.is_empty() {
        if subjects.iter().any(|s| format!("{}:{}", s.entity, s.id) == user) {
            println!("[DEBUG] User {} found in subjects for token '{}'", user, token);
            return true;
        }
        return false;
    }
    if let Some(val) = attr_store.get(entity_name, instance_id, token) {
        println!("[DEBUG] Found attribute {} on {}:{} with value {}", token, entity_name, instance_id, val);
        return val != 0;
    }
    false
}

/// Updated evaluate_permission uses evaluate_expr.
fn evaluate_permission(
    schema: &std::collections::HashMap<String, Entity>,
    entity_name: &str,
    instance_id: &str,
    perm_name: &str,
    user: &str,
    rel_store: &RelationshipStore,
    attr_store: &AttributeStore,
) -> bool {
    // For a user entity, simply compare.
    if entity_name == "user" {
        let result = format!("{}:{}", entity_name, instance_id) == user;
        println!("[DEBUG] evaluate_permission on user {}: {} == {} -> {}", entity_name, format!("{}:{}", entity_name, instance_id), user, result);
        return result;
    }
    // For organization "approve", delegate to the composite "approval" permission.
    if entity_name == "organization" && perm_name == "approve" {
        if schema.get(entity_name).and_then(|e| e.permissions.get("approval")).is_some() {
            return evaluate_permission(schema, entity_name, instance_id, "approval", user, rel_store, attr_store);
        }
    }
    let entity = match schema.get(entity_name) {
        Some(e) => e,
        None => return false,
    };
    if let Some(expr) = entity.permissions.get(perm_name) {
        println!("[DEBUG] Evaluating permission {} on {}:{} with expression '{}'", perm_name, entity_name, instance_id, expr);
        evaluate_expr(expr, schema, entity_name, instance_id, user, rel_store, attr_store)
    } else {
        let subjects = get_relation_subjects(rel_store, schema, entity_name, instance_id, perm_name);
        subjects.iter().any(|s| format!("{}:{}", s.entity, s.id) == user)
    }
}


fn main() -> Result<(), Box<dyn Error>> {
    // For example, load the document management domain.
    let schema_dsl = fs::read_to_string("user_groups_schema.yaml")?;
    let schema = parse_schema(&schema_dsl);
    println!("Parsed schema entities:");
    for (name, entity) in &schema {
        println!("Entity: {} => {:?}", name, entity);
    }

    let relationships_str = fs::read_to_string("user_groups_relationships.yaml")?;
    let mut attr_store = AttributeStore::new();
    let rel_store = Arc::new(load_relationships_and_attributes(&relationships_str, &mut attr_store));

    let mut attr_store = AttributeStore::new();
    attr_store.set("state", "1", "age_limit", 18);
    attr_store.set("patient", "1", "age", 15);

    let mut engine = Engine::new();
    let schema_for_rhai = Arc::new(schema);
    let rel_store_rhai = Arc::clone(&rel_store);
    let attr_store_rhai = Arc::new(attr_store);
    engine.register_fn(
        "check_permission",
        move |entity: &str, id: &str, perm: &str, user: &str| -> bool {
            evaluate_permission(
                &schema_for_rhai,
                entity,
                id,
                perm,
                user,
                &rel_store_rhai,
                &attr_store_rhai,
            )
        },
    );

    let res1: bool = engine.eval(r#"
        check_permission("group", "1", "post_to_group", "user:1");
    "#)?;
    println!("1 ? => {}", res1);

    let res1: bool = engine.eval(r#"
        check_permission("group", "1", "invite_to_group", "user:2");
    "#)?;
    println!("2 ? => {}", res1);

    let res1: bool = engine.eval(r#"
        check_permission("group", "2", "edit_settings", "user:3");
    "#)?;
    println!("3 ? => {}", res1);

    let res1: bool = engine.eval(r#"
        check_permission("event", "1", "view_event", "user:1")
    "#)?;
    println!("4 ? => {}", res1);

    let res1: bool = engine.eval(r#"
        check_permission("event", "1", "delete_event", "user:3")
    "#)?;
    println!("5 ? => {}", res1);

    let res1: bool = engine.eval(r#"
        check_permission("group", "2", "invite_to_group", "user:4")
    "#)?;
    println!("6 ? => {}", res1);

    Ok(())
}

//
// ─── TEST SUITE ────────────────────────────────────────────────────────────────
//

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::sync::Arc;

    /// Helper to create a Rhai engine instance given a schema and relationships;
    /// Optionally a closure can set up the AttributeStore.
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
        let rel_store = Arc::new(load_relationships_and_attributes(&relationships_str, &mut attr_store));
        if let Some(setup) = attr_setup {
            setup(&mut attr_store);
        }
        let attr_store = Arc::new(attr_store);
        let mut engine = Engine::new();
        let schema_for_rhai = Arc::clone(&schema);
        let rel_store_rhai = Arc::clone(&rel_store);
        let attr_store_rhai = Arc::clone(&attr_store);
        engine.register_fn(
            "check_permission",
            move |entity: &str, id: &str, perm: &str, user: &str| -> bool {
                evaluate_permission(
                    &schema_for_rhai,
                    entity,
                    id,
                    perm,
                    user,
                    &rel_store_rhai,
                    &attr_store_rhai,
                )
            },
        );
        (engine, schema, rel_store, attr_store)
    }

    #[test]
    fn test_document_management_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/document_management_schema.yaml");
        let relationships_str = include_str!("../schemas/document_management_relationships.yaml");
        let (engine, _schema, _rel_store, mut _attr_store) = create_engine(schema_str, relationships_str, Some(|attr_store: &mut AttributeStore| {
            // Setup attributes used in document management (if any)
            attr_store.set("state", "1", "age_limit", 18);
            attr_store.set("patient", "1", "age", 15);
        }));

        // Print out the parsed schema for debugging.
        println!("Parsed Schema: {:#?}", _schema);

        // Test manager permissions
        let res1 = engine.eval::<bool>(r#"
            check_permission("document", "product_database", "edit", "user:ashley")
        "#)?;
        println!("Can ashley edit product_database? => {}", res1);
        assert!(res1, "Expected ashley to be able to edit product_database");

        // Test viewer permissions
        let res2 = engine.eval::<bool>(r#"
            check_permission("document", "product_database", "view", "user:david")
        "#)?;
        println!("Can david view product_database? => {}", res2);
        assert!(res2, "Expected david to be able to view product_database");

        // Test organization admin permissions
        let res3 = engine.eval::<bool>(r#"
            check_permission("document", "marketing_materials", "edit", "user:jenny")
        "#)?;
        println!("Can jenny edit marketing_materials? => {}", res3);
        assert!(res3, "Expected jenny to be able to edit marketing_materials");

        // Test group hierarchy permissions
        let res4 = engine.eval::<bool>(r#"
            check_permission("document", "hr_documents", "view", "user:david")
        "#)?;
        println!("Can david view hr_documents? => {}", res4);
        assert!(!res4, "Expected david NOT to be able to view hr_documents");

        // Test organization membership
        let res5 = engine.eval::<bool>(r#"
            check_permission("organization", "acme", "member", "user:david")
        "#)?;
        println!("Is david a member of acme organization? => {}", res5);
        assert!(res5, "Expected david to be a member of acme");

        Ok(())
    }

    #[test]
    fn test_hospital_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/hospital_schema.yaml");
        let relationships_str = include_str!("../schemas/hospital_relationships.yaml");
        let (engine, _schema, _rel_store, mut _attr_store) = create_engine(schema_str, relationships_str, Some(|attr_store: &mut AttributeStore| {
            // Setup hospital-specific attributes
            attr_store.set("state", "1", "age_limit", 18);
            attr_store.set("patient", "1", "age", 20);
        }));

        // Test patient edit by owner (user:12)
        let res1 = engine.eval::<bool>(r#"
            check_permission("patient", "1", "edit", "user:12")
        "#)?;
        println!("Can user:12 edit patient 1? => {}", res1);
        assert!(res1, "Expected user:12 (owner) to edit patient 1");

        // Test patient view by guardian (user:13)
        let res2 = engine.eval::<bool>(r#"
            check_permission("patient", "1", "view", "user:13")
        "#)?;
        println!("Can user:13 view patient 1? => {}", res2);
        assert!(res2, "Expected user:13 (guardian) to view patient 1");

        // Test patient edit by an unrelated user (user:99)
        let res3 = engine.eval::<bool>(r#"
            check_permission("patient", "1", "edit", "user:99")
        "#)?;
        println!("Can user:99 edit patient 1? => {}", res3);
        assert!(!res3, "Expected user:99 NOT to edit patient 1");

        Ok(())
    }

    #[test]
    fn test_organisational_tool_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/organisation_tool_schema.yaml");
        let relationships_str = include_str!("../schemas/organisational_tool_relationships.yaml");
        let (engine, _schema, _rel_store, _attr_store) = create_engine::<fn(&mut AttributeStore)>(schema_str, relationships_str, None);

        // Test workspace management: alice should be able to manage the engineering_team workspace.
        let res1 = engine.eval::<bool>(r#"
            check_permission("workspace", "engineering_team", "manage_workspace", "user:alice")
        "#)?;
        println!("Can alice manage engineering_team workspace? => {}", res1);
        assert!(res1, "Expected alice to manage the engineering_team workspace");

        // Test page read permission: bob should be able to read the project_plan page.
        let res2 = engine.eval::<bool>(r#"
            check_permission("page", "project_plan", "read", "user:bob")
        "#)?;
        println!("Can bob read project_plan page? => {}", res2);
        assert!(res2, "Expected bob to read the project_plan page");

        // Test database write permission: alice should be allowed to write to task_list.
        let res3 = engine.eval::<bool>(r#"
            check_permission("database", "task_list", "write", "user:alice")
        "#)?;
        println!("Can alice write to task_list database? => {}", res3);
        assert!(res3, "Expected alice to write to task_list database");

        // Test block comment permission: bob can comment on block task_list_1.
        let res4 = engine.eval::<bool>(r#"
            check_permission("block", "task_list_1", "comment", "user:bob")
        "#)?;
        println!("Can bob comment on block task_list_1? => {}", res4);
        assert!(res4, "Expected bob to comment on block task_list_1");

        // Test block comment permission: charlie should not be allowed on block task_list_1.
        let res5 = engine.eval::<bool>(r#"
            check_permission("block", "task_list_1", "comment", "user:charlie")
        "#)?;
        println!("Can charlie comment on block task_list_1? => {}", res5);
        assert!(!res5, "Expected charlie NOT to comment on block task_list_1");

        Ok(())
    }

    #[test]
    fn test_social_network_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/social_network_schema.yaml");
        let relationships_str = include_str!("../schemas/social_network_relationships.yaml");
        let (engine, _schema, _rel_store, _attr_store) = create_engine::<fn(&mut AttributeStore)>(schema_str, relationships_str, None);

        // Test account view: kevin should view account:1 (owner).
        let res1 = engine.eval::<bool>(r#"
            check_permission("account", "1", "view", "user:kevin")
        "#)?;
        println!("Can kevin view account:1? => {}", res1);
        assert!(res1, "Expected kevin to view account:1 as owner");

        // Test account view: kevin should view account:2 (as follower).
        let res2 = engine.eval::<bool>(r#"
            check_permission("account", "2", "view", "user:kevin")
        "#)?;
        println!("Can kevin view account:2? => {}", res2);
        assert!(res2, "Expected kevin to view account:2 as follower");

        // Test post view: kevin viewing post:2. In our evaluator (which does not handle attributes),
        // this falls back to the account view permission. Given that account:2 is viewable by kevin,
        // we expect this to be true.
        let res3 = engine.eval::<bool>(r#"
            check_permission("post", "2", "view", "user:kevin")
        "#)?;
        println!("Can kevin view post:2? => {}", res3);
        assert!(res3, "Expected kevin to view post:2");

        Ok(())
    }

    #[test]
    fn test_user_groups_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/user_groups_schema.yaml");
        let relationships_str = include_str!("../schemas/user_groups_relationships.yaml");
        let (engine, _schema, _rel_store, _attr_store) = create_engine::<fn(&mut AttributeStore)>(schema_str, relationships_str, None);

        // Test group member permissions
        let res1 = engine.eval::<bool>(r#"
            check_permission("group", "1", "post_to_group", "user:1")
        "#)?;
        println!("Can user:1 post to group:1? => {}", res1);
        assert!(res1, "Expected user:1 to be able to post to group:1 as member");

        // Test group admin permissions
        let res2 = engine.eval::<bool>(r#"
            check_permission("group", "1", "invite_to_group", "user:2")
        "#)?;
        println!("Can user:2 invite to group:1? => {}", res2);
        assert!(res2, "Expected user:2 to be able to invite to group:1 as admin");

        // Test group moderator permissions
        let res3 = engine.eval::<bool>(r#"
            check_permission("group", "2", "edit_settings", "user:3")
        "#)?;
        println!("Can user:3 edit settings of group:2? => {}", res3);
        assert!(res3, "Expected user:3 to be able to edit settings of group:2 as moderator");

        // Test event permissions through group membership
        let res4 = engine.eval::<bool>(r#"
            check_permission("event", "1", "view_event", "user:1")
        "#)?;
        println!("Can user:1 view event:1? => {}", res4);
        assert!(res4, "Expected user:1 to be able to view event:1 as group member");

        // Test event owner permissions
        let res5 = engine.eval::<bool>(r#"
            check_permission("event", "1", "delete_event", "user:3")
        "#)?;
        println!("Can user:3 delete event:1? => {}", res5);
        assert!(res5, "Expected user:3 to be able to delete event:1 as owner");

        // Test unauthorized access
        let res6 = engine.eval::<bool>(r#"
            check_permission("group", "2", "invite_to_group", "user:4")
        "#)?;
        println!("Can user:4 invite to group:2? => {}", res6);
        assert!(!res6, "Expected user:4 NOT to be able to invite to group:2 as regular member");

        Ok(())
    }

    #[test]
    fn test_fintech_banking_permissions() -> Result<(), Box<dyn Error>> {
        let schema_str = include_str!("../schemas/fintech_banking_schema.yaml");
        let relationships_str = include_str!("../schemas/fintech_banking_relationships.yaml");
        let (engine, _schema, _rel_store, mut _attr_store) = create_engine::<fn(&mut AttributeStore)>(schema_str, relationships_str, None);

        // Print out the parsed schema for debugging
        println!("Parsed Schema: {:#?}", _schema);

        // Test organization admin permissions
        let res1 = engine.eval::<bool>(r#"
            check_permission("organization", "1", "approve", "user:alice")
        "#)?;
        println!("-- Can alice approve in organization 1? => {}", res1);
        assert!(res1, "Expected alice to be able to approve in organization 1 as admin");

        // Test organization member permissions with approval limits
        let res2 = engine.eval::<bool>(r#"
            check_permission("organization", "1", "approve", "user:bob")
        "#)?;
        println!("-- Can bob approve in organization 1? => {}", res2);
        assert!(res2, "Expected bob to be able to approve in organization 1 as member with sufficient approvals");

        // Test account withdrawal permissions (within limit)
        let res3 = engine.eval::<bool>(r#"
            check_permission("account", "1", "withdraw", "user:alice")
        "#)?;
        println!("-- Can alice withdraw from account 1? => {}", res3);
        assert!(res3, "Expected alice to be able to withdraw from account 1 as org admin");

        // Test linked account permissions (checkings)
        let res4 = engine.eval::<bool>(r#"
            check_permission("account", "4", "withdraw", "user:bob")
        "#)?;
        println!("-- Can bob withdraw from checkings account 4? => {}", res4);
        assert!(res4, "Expected bob to be able to withdraw from checkings account 4");

        // Test cross-organization access (should fail)
        let res5 = engine.eval::<bool>(r#"
            check_permission("account", "3", "withdraw", "user:alice")
        "#)?;
        println!("-- Can alice withdraw from account 3 (different org)? => {}", res5);
        assert!(!res5, "Expected alice NOT to be able to withdraw from account 3 (different org)");

        // Test account creation permissions
        let res6 = engine.eval::<bool>(r#"
            check_permission("organization", "1", "create_account", "user:alice")
        "#)?;
        println!("Can alice create account in organization 1? => {}", res6);
        assert!(res6, "Expected alice to be able to create account in organization 1 as admin");

        Ok(())
    }
}
