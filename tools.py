
# First, let's create the tools following the same pattern
def semantic_search_anything(query: str, limit: int = 10) -> dict:
    """Semantic search across all nodes in the graph using vector embeddings.

    Args:
        query: Natural language query to search for (e.g., "expensive electronics", "car parts")
        limit: Number of results to return (default: 10)

    Returns:
        Dictionary with status and search results.
        Success: {"status": "success", "results": [...]}
        Error: {"status": "error", "error_message": "Search failed"}
    """
    try:
        # This would use your DynamicVectorManager internally
        query_embedding = vector_manager.embedding_manager.get_embedding(query)
        
        # Search across all labels with embeddings
        all_results = []
        for label in vector_manager.schema.keys():
            search_query = f"""
            CALL db.index.vector.queryNodes('{label.lower()}_embeddings', $limit, $embedding)
            YIELD node, score
            WHERE node:{label}
            RETURN node.id as node_id, 
                   labels(node) as labels,
                   properties(node) as properties,
                   score
            ORDER BY score DESC
            """
            
            try:
                results = graph.query(search_query, {
                    "embedding": query_embedding,
                    "limit": limit
                })
                all_results.extend(results)
            except Exception:
                continue
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return {
            "status": "success", 
            "results": all_results[:limit],
            "total_matches": len(all_results)
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Search failed: {str(e)}"
        }

def discover_graph_structure() -> dict:
    """Discover and return the complete graph structure including nodes and relationships.

    Returns:
        Dictionary with status and graph structure information.
        Success: {"status": "success", "node_statistics": [...], "relationships": [...]}
        Error: {"status": "error", "error_message": "Discovery failed"}
    """
    try:
        # Get node counts
        node_counts = []
        for label in vector_manager.schema.keys():
            count_query = f"MATCH (n:{label}) RETURN count(n) as count"
            result = graph.query(count_query)
            node_counts.append({"label": label, "count": result[0]['count']})
        
        # Get relationships
        rel_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
        relationships = graph.query(rel_query)
        
        return {
            "status": "success",
            "node_statistics": node_counts,
            "relationships": [r['relationshipType'] for r in relationships],
            "total_node_types": len(node_counts),
            "total_relationship_types": len(relationships)
        }
    except Exception as e:
        return {
            "status": "error", 
            "error_message": f"Graph discovery failed: {str(e)}"
        }

def explore_from_starting_point(starting_point: str) -> dict:
    """Explore the graph starting from a specific concept or search term.

    Args:
        starting_point: What to start exploring from (e.g., "receipts", "cars", "electronics")

    Returns:
        Dictionary with status and exploration results.
        Success: {"status": "success", "matched_node": {...}, "connections": [...]}
        Error: {"status": "error", "error_message": "Exploration failed"}
    """
    try:
        # First try semantic search
        search_result = semantic_search_anything(starting_point, limit=1)
        
        if search_result["status"] == "success" and search_result["results"]:
            first_result = search_result["results"][0]
            node_id = first_result['node_id']
            
            # Find connections from this node
            connections_query = """
            MATCH (n {id: $node_id})-[r]-(connected)
            RETURN type(r) as relationship_type, 
                   labels(connected) as connected_labels,
                   connected.id as connected_id,
                   connected.name as connected_name
            LIMIT 10
            """
            
            connections = graph.query(connections_query, {"node_id": node_id})
            
            return {
                "status": "success",
                "starting_point": starting_point,
                "matched_node": first_result,
                "connections": connections,
                "total_connections": len(connections)
            }
        else:
            return {
                "status": "error",
                "error_message": f"No matches found for: {starting_point}"
            }
    except Exception as e:
        return {
            "status": "error",
            "error_message": f"Exploration failed: {str(e)}"
        }
