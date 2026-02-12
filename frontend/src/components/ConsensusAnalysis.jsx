import "./ConsensusAnalysis.css";

export default function ConsensusAnalysis({ consensus }) {
  // Debug logging
  console.log("ConsensusAnalysis received:", consensus);

  // Don't render if consensus is null or undefined
  if (!consensus) {
    console.log("No consensus data");
    return null;
  }

  // Show error state if there was an error
  if (consensus.error && !consensus.best_answer) {
    console.log("Consensus error:", consensus.error);
    return null; // Silently skip if consensus failed completely
  }

  const {
    best_answer,
    model_consensus_scores,
    merged_answer,
    confidence,
    models_agreed,
    total_models,
    disagreement_detected,
    low_consensus,
    warning_message,
    hallucination_risk,
  } = consensus;
  console.log("Rendering consensus with:", {
    best_answer,
    model_consensus_scores,
    merged_answer,
    confidence,
    models_agreed,
    total_models,
    disagreement_detected,
    low_consensus,
    warning_message,
    hallucination_risk,
  });

  // Sort models by consensus score (highest first)
  const sortedScores = Object.entries(model_consensus_scores || {}).sort(
    ([, a], [, b]) => b - a,
  );

  // Determine confidence level and color
  const getConfidenceLevel = (score) => {
    if (score >= 0.8) return { level: "High", color: "#10b981" };
    if (score >= 0.6) return { level: "Medium", color: "#f59e0b" };
    return { level: "Low", color: "#ef4444" };
  };

  const confidenceInfo =
    confidence !== undefined ? getConfidenceLevel(confidence) : null;

  return (
    <div className="consensus-analysis">
      <div className="consensus-header">
        <h3>🎯 Semantic Consensus Analysis</h3>
        <p className="consensus-description">
          Using AI embeddings to measure response similarity
        </p>
      </div>

      {/* Disagreement Warning */}
      {(disagreement_detected || low_consensus) && (
        <div className="disagreement-warning">
          <div className="warning-icon">⚠️</div>
          <div className="warning-content">
            <div className="warning-title">
              Low Confidence — Models Disagree
            </div>
            <div className="warning-message">
              {warning_message ||
                "Models have varying perspectives on this topic."}
            </div>
            <div className="warning-actions">
              <div className="action-suggestion">
                💡 <strong>Suggestions:</strong>
              </div>
              <ul className="suggestion-list">
                <li>Rephrase your question with more specific details</li>
                <li>Provide additional context or constraints</li>
                <li>Break down complex questions into simpler parts</li>
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Hallucination Risk Display */}
      {hallucination_risk && (
        <div
          className={`hallucination-risk-section risk-${hallucination_risk.risk_level?.toLowerCase() || "unknown"}`}
        >
          <div className="risk-header">
            <span className="risk-icon">
              {hallucination_risk.risk_icon || "⚠️"}
            </span>
            <span className="risk-title">
              Hallucination Risk: {hallucination_risk.risk_level || "UNKNOWN"}
            </span>
          </div>
          <div className="risk-message">{hallucination_risk.risk_message}</div>

          {/* Contributing Factors */}
          {hallucination_risk.contributing_factors && (
            <div className="risk-factors">
              <div className="factors-title">Contributing Factors:</div>
              <div className="factors-grid">
                {hallucination_risk.contributing_factors.entropy_risk !==
                  undefined && (
                  <div className="factor-item">
                    <span className="factor-label">Entropy:</span>
                    <span className="factor-value">
                      {(
                        hallucination_risk.contributing_factors.entropy_risk *
                        100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                )}
                {hallucination_risk.contributing_factors.similarity_risk !==
                  undefined && (
                  <div className="factor-item">
                    <span className="factor-label">Similarity:</span>
                    <span className="factor-value">
                      {(
                        hallucination_risk.contributing_factors
                          .similarity_risk * 100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                )}
                {hallucination_risk.contributing_factors.variance_risk !==
                  undefined && (
                  <div className="factor-item">
                    <span className="factor-label">Variance:</span>
                    <span className="factor-value">
                      {(
                        hallucination_risk.contributing_factors.variance_risk *
                        100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                )}
                {hallucination_risk.contributing_factors.contradiction_risk !==
                  undefined && (
                  <div className="factor-item">
                    <span className="factor-label">Contradictions:</span>
                    <span className="factor-value">
                      {(
                        hallucination_risk.contributing_factors
                          .contradiction_risk * 100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                )}
                {hallucination_risk.contributing_factors.inconsistency_risk !==
                  undefined && (
                  <div className="factor-item">
                    <span className="factor-label">Inconsistencies:</span>
                    <span className="factor-value">
                      {(
                        hallucination_risk.contributing_factors
                          .inconsistency_risk * 100
                      ).toFixed(1)}
                      %
                    </span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Detected Contradictions */}
          {hallucination_risk.contradictions &&
            hallucination_risk.contradictions.length > 0 && (
              <div className="contradictions-section">
                <div className="contradictions-title">
                  🚨 Detected Contradictions:
                </div>
                <ul className="contradictions-list">
                  {hallucination_risk.contradictions.map(
                    (contradiction, idx) => (
                      <li key={idx} className="contradiction-item">
                        <strong>{contradiction.models.join(" vs ")}:</strong>{" "}
                        {contradiction.type}
                      </li>
                    ),
                  )}
                </ul>
              </div>
            )}

          {/* Factual Inconsistencies */}
          {hallucination_risk.inconsistencies &&
            hallucination_risk.inconsistencies.length > 0 && (
              <div className="inconsistencies-section">
                <div className="inconsistencies-title">
                  ⚡ Factual Inconsistencies:
                </div>
                <ul className="inconsistencies-list">
                  {hallucination_risk.inconsistencies.map(
                    (inconsistency, idx) => (
                      <li key={idx} className="inconsistency-item">
                        <strong>{inconsistency.models.join(" vs ")}:</strong>{" "}
                        {inconsistency.type}
                      </li>
                    ),
                  )}
                </ul>
              </div>
            )}
        </div>
      )}

      {/* Confidence Score Display */}
      {confidenceInfo && (
        <div className="confidence-section">
          <div className="confidence-card">
            <div className="confidence-label">Overall Confidence</div>
            <div className="confidence-display">
              <div
                className="confidence-percentage"
                style={{ color: confidenceInfo.color }}
              >
                {(confidence * 100).toFixed(1)}%
              </div>
              <div
                className="confidence-level"
                style={{ color: confidenceInfo.color }}
              >
                {confidenceInfo.level}
              </div>
            </div>
            {models_agreed !== undefined && total_models !== undefined && (
              <div className="agreement-stats">
                <span className="stat-icon">🤝</span>
                <span>
                  {models_agreed} of {total_models} models in high agreement
                </span>
              </div>
            )}
            <div className="confidence-bar-full">
              <div
                className="confidence-bar-fill"
                style={{
                  width: `${confidence * 100}%`,
                  background: confidenceInfo.color,
                }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* Best Answer */}
      {best_answer && (
        <div className="best-answer-section">
          <div className="section-title">
            <span className="trophy-icon">🏆</span>
            <span>Highest Agreement Answer</span>
          </div>
          <div className="best-answer-card">
            <div className="best-answer-header">
              <span className="model-name">{best_answer.model}</span>
              <span className="consensus-badge">
                {(best_answer.consensus_score * 100).toFixed(1)}% consensus
              </span>
            </div>
            <div className="consensus-bar">
              <div
                className="consensus-fill"
                style={{ width: `${best_answer.consensus_score * 100}%` }}
              ></div>
            </div>
          </div>
        </div>
      )}

      {/* All Model Scores */}
      {sortedScores.length > 0 && (
        <div className="scores-section">
          <div className="section-title">Model Consensus Scores</div>
          <div className="scores-grid">
            {sortedScores.map(([model, score]) => (
              <div key={model} className="score-item">
                <div className="score-model">{model}</div>
                <div className="score-bar-container">
                  <div
                    className="score-bar"
                    style={{ width: `${score * 100}%` }}
                  ></div>
                </div>
                <div className="score-value">{(score * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Merged Answer */}
      {merged_answer && (
        <div className="merged-answer-section">
          <div className="section-title">
            <span className="merge-icon">🔀</span>
            <span>Synthesized Consensus Answer</span>
          </div>
          <div className="merged-answer-content">{merged_answer}</div>
        </div>
      )}
    </div>
  );
}
