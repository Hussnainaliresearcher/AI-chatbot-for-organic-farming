import pandas as pd
import numpy as np
from openai import OpenAI
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import re

class ComprehensiveEvaluator:
    def __init__(self):
        """Initialize with OpenAI client for embeddings and LLM judge"""
        self.client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        self.all_knowledge_base = []  # Will store all answers from Excel for hallucination check
    
    def load_full_knowledge_base(self, file_path="zone wise data.xlsx", sheet_name="agro zones and organic farming"):
        """Load complete knowledge base for hallucination detection"""
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            answers = df.iloc[:, 1].dropna().tolist()  # Second column = answers
            self.all_knowledge_base = [str(ans).lower() for ans in answers]
            print(f"✅ Loaded {len(self.all_knowledge_base)} knowledge base entries for hallucination check")
        except Exception as e:
            print(f"⚠️  Could not load full knowledge base: {e}")
    
    def get_embeddings(self, texts):
        """Get OpenAI embeddings for similarity comparison"""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            return np.array([item.embedding for item in response.data])
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return None
    
    def calculate_similarity(self, answer1, answer2):
        """Calculate similarity between two answers (0-1 score)"""
        embeddings = self.get_embeddings([answer1, answer2])
        
        if embeddings is None:
            return 0.0
        
        similarity = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return similarity
    
    def calculate_relevance(self, question, answer):
        """Calculate how relevant the answer is to the question (0-1 score)"""
        embeddings = self.get_embeddings([question, answer])
        
        if embeddings is None:
            return 0.0
        
        relevance = cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0][0]
        
        return relevance
    
    def calculate_completeness(self, chatbot_answer, actual_answer):
        """Calculate completeness by comparing answer lengths and key information coverage"""
        # Length ratio (chatbot vs actual)
        chatbot_len = len(chatbot_answer.split())
        actual_len = len(actual_answer.split())
        
        if actual_len == 0:
            return 0.0
        
        length_ratio = min(chatbot_len / actual_len, 1.0)  # Cap at 1.0
        
        # Key terms coverage (simple approach)
        actual_words = set(actual_answer.lower().split())
        chatbot_words = set(chatbot_answer.lower().split())
        
        # Remove common words
        common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'must', 'a', 'an', 'this', 'that', 'these', 'those'}
        actual_words = actual_words - common_words
        chatbot_words = chatbot_words - common_words
        
        if len(actual_words) == 0:
            key_coverage = 1.0
        else:
            key_coverage = len(actual_words.intersection(chatbot_words)) / len(actual_words)
        
        # Combine length ratio and key coverage (weighted average)
        completeness = (length_ratio * 0.4) + (key_coverage * 0.6)
        
        return completeness
    
    def extract_key_facts(self, text):
        """Extract key facts/entities from text for hallucination checking"""
        # Simple fact extraction - you can enhance this
        facts = []
        
        # Extract numbers with units (e.g., "2kg", "30 days", "5 liters")
        numbers = re.findall(r'\d+\.?\d*\s*(?:kg|grams?|g|liters?|l|days?|weeks?|months?|years?|cm|meters?|m|%|percent)', text.lower())
        facts.extend(numbers)
        
        # Extract specific terms (fertilizer names, methods, etc.)
        # You can add more domain-specific patterns here
        specific_terms = re.findall(r'\b(?:neem|vermi|compost|urea|potash|phosphate|organic|bio|natural)\w*\b', text.lower())
        facts.extend(specific_terms)
        
        # Extract quoted or capitalized terms that might be product names
        products = re.findall(r'\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\b', text)
        facts.extend([p.lower() for p in products if len(p) > 3])
        
        return list(set(facts))  # Remove duplicates
    
    def check_fact_in_knowledge_base(self, fact):
        """Check if a fact exists in the knowledge base"""
        fact = fact.lower().strip()
        
        for kb_answer in self.all_knowledge_base:
            if fact in kb_answer:
                return True
        return False
    
    def calculate_hallucination_rate_method1(self, chatbot_answer):
        """Method 1: Fact-based hallucination detection"""
        if not self.all_knowledge_base:
            return 0.0  # Can't calculate without knowledge base
        
        facts = self.extract_key_facts(chatbot_answer)
        
        if not facts:
            return 0.0  # No specific facts to check
        
        hallucinated_facts = 0
        for fact in facts:
            if not self.check_fact_in_knowledge_base(fact):
                hallucinated_facts += 1
        
        hallucination_rate = hallucinated_facts / len(facts)
        return hallucination_rate
    
    def calculate_hallucination_rate_method2(self, chatbot_answer, actual_answer):
        """Method 2: LLM-as-Judge for hallucination detection"""
        try:
            prompt = f"""You are evaluating if a chatbot answer contains hallucinated (made-up) information.

Reference Answer (Ground Truth): "{actual_answer}"

Chatbot Answer: "{chatbot_answer}"

Task: Compare the chatbot answer with the reference answer. Identify if the chatbot added any specific facts, numbers, names, or details that are NOT present or implied in the reference answer.

Respond with ONLY a number between 0.0 and 1.0 where:
- 0.0 = No hallucination (all information is supported by reference)
- 1.0 = High hallucination (most information is made-up)

Consider hallucination:
- Specific numbers not in reference (e.g., "2kg" when reference says "some")
- Product names not mentioned in reference
- Specific timeframes not in reference
- Technical details not supported by reference

Score:"""

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Cheaper model for evaluation
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=10
            )
            
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Ensure 0-1 range
            
        except Exception as e:
            print(f"Error in LLM hallucination check: {e}")
            return 0.0
    
    def calculate_hallucination_rate(self, chatbot_answer, actual_answer):
        """Combined hallucination rate using both methods"""
        method1_score = self.calculate_hallucination_rate_method1(chatbot_answer)
        method2_score = self.calculate_hallucination_rate_method2(chatbot_answer, actual_answer)
        
        # Average both methods (you can adjust weights)
        combined_score = (method1_score * 0.4) + (method2_score * 0.6)
        return combined_score
    
    def evaluate_answers(self, chatbot_file="chatbot_result.xlsx", actual_file="actual_answers.xlsx"):
        """
        Comprehensive evaluation: Accuracy, Relevance, Completeness, Hallucination
        """
        try:
            # Load knowledge base for hallucination detection
            self.load_full_knowledge_base()
            
            # Load both files
            chatbot_df = pd.read_excel(chatbot_file)
            actual_df = pd.read_excel(actual_file)
            
            print("✅ Files loaded successfully!")
            print(f"Chatbot file: {len(chatbot_df)} rows")
            print(f"Actual file: {len(actual_df)} rows")
            
            # Assume columns are: Question, Answer (adjust if different)
            chatbot_questions = chatbot_df.iloc[:, 0].tolist()  # First column
            chatbot_answers = chatbot_df.iloc[:, 1].tolist()    # Second column
            
            actual_questions = actual_df.iloc[:, 0].tolist()    # First column  
            actual_answers = actual_df.iloc[:, 1].tolist()      # Second column
            
            # Check if same number of questions
            if len(chatbot_answers) != len(actual_answers):
                print("⚠️  Warning: Different number of Q&A pairs in files")
                min_len = min(len(chatbot_answers), len(actual_answers))
                chatbot_answers = chatbot_answers[:min_len]
                actual_answers = actual_answers[:min_len]
                chatbot_questions = chatbot_questions[:min_len]
            
            print(f"\n🔍 Evaluating {len(chatbot_answers)} answer pairs...")
            print("="*80)
            
            results = []
            total_accuracy = 0
            total_relevance = 0
            total_completeness = 0
            total_hallucination = 0
            
            # Evaluate each answer pair
            for i in range(len(chatbot_answers)):
                print(f"\n📝 Question {i+1}:")
                print(f"Q: {str(chatbot_questions[i])[:80]}...")
                
                chatbot_ans = str(chatbot_answers[i])
                actual_ans = str(actual_answers[i])
                
                # Calculate all metrics
                accuracy = self.calculate_similarity(chatbot_ans, actual_ans)
                relevance = self.calculate_relevance(chatbot_questions[i], chatbot_ans)
                completeness = self.calculate_completeness(chatbot_ans, actual_ans)
                hallucination = self.calculate_hallucination_rate(chatbot_ans, actual_ans)
                
                # Convert to percentages
                accuracy_pct = round(accuracy * 100, 1)
                relevance_pct = round(relevance * 100, 1)
                completeness_pct = round(completeness * 100, 1)
                hallucination_pct = round(hallucination * 100, 1)
                
                # Calculate overall score (lower hallucination is better)
                overall_score = (accuracy + relevance + completeness + (1 - hallucination)) / 4
                
                total_accuracy += accuracy
                total_relevance += relevance
                total_completeness += completeness
                total_hallucination += hallucination
                
                # Store result
                result = {
                    'question_no': i+1,
                    'question': chatbot_questions[i],
                    'chatbot_answer': chatbot_ans,
                    'actual_answer': actual_ans,
                    'accuracy_score': round(accuracy, 3),
                    'accuracy_percentage': accuracy_pct,
                    'relevance_score': round(relevance, 3),
                    'relevance_percentage': relevance_pct,
                    'completeness_score': round(completeness, 3),
                    'completeness_percentage': completeness_pct,
                    'hallucination_score': round(hallucination, 3),
                    'hallucination_percentage': hallucination_pct,
                    'overall_score': round(overall_score, 3),
                    'overall_percentage': round(overall_score * 100, 1)
                }
                results.append(result)
                
                # Print result with all metrics
                print(f"🤖 Chatbot: {chatbot_ans[:60]}...")
                print(f"✅ Actual:  {actual_ans[:60]}...")
                print(f"🎯 Accuracy: {accuracy_pct}% | 🔗 Relevance: {relevance_pct}% | 📝 Completeness: {completeness_pct}% | 🚫 Hallucination: {hallucination_pct}%")
                
                # Overall interpretation
                if overall_score > 0.8:
                    print("   → Overall: Excellent! 🌟")
                elif overall_score > 0.6:
                    print("   → Overall: Good 👍")
                elif overall_score > 0.4:
                    print("   → Overall: Fair ⚠️")
                else:
                    print("   → Overall: Needs Improvement 🔴")
            
            # Calculate overall results for all metrics
            avg_accuracy = (total_accuracy / len(results)) * 100
            avg_relevance = (total_relevance / len(results)) * 100
            avg_completeness = (total_completeness / len(results)) * 100
            avg_hallucination = (total_hallucination / len(results)) * 100
            avg_overall = (sum(r['overall_score'] for r in results) / len(results)) * 100
            
            print(f"\n" + "="*80)
            print(f"📊 COMPREHENSIVE EVALUATION RESULTS:")
            print(f"Total Questions Evaluated: {len(results)}")
            print(f"🎯 Average Accuracy:      {round(avg_accuracy, 1)}%")
            print(f"🔗 Average Relevance:     {round(avg_relevance, 1)}%")
            print(f"📝 Average Completeness:  {round(avg_completeness, 1)}%")
            print(f"🚫 Average Hallucination: {round(avg_hallucination, 1)}% (Lower is Better)")
            print(f"⭐ Overall Performance:   {round(avg_overall, 1)}%")
            
            # Score breakdown for overall performance
            excellent = sum(1 for r in results if r['overall_score'] > 0.8)
            good = sum(1 for r in results if 0.6 <= r['overall_score'] <= 0.8)
            fair = sum(1 for r in results if 0.4 <= r['overall_score'] <= 0.6)
            poor = sum(1 for r in results if r['overall_score'] < 0.4)
            
            print(f"\n📈 Overall Performance Distribution:")
            print(f"🌟 Excellent (>80%): {excellent} answers")
            print(f"👍 Good (60-80%):    {good} answers")
            print(f"⚠️  Fair (40-60%):    {fair} answers")
            print(f"🔴 Poor (<40%):      {poor} answers")
            
            # Hallucination analysis
            low_halluc = sum(1 for r in results if r['hallucination_score'] < 0.2)
            med_halluc = sum(1 for r in results if 0.2 <= r['hallucination_score'] <= 0.5)
            high_halluc = sum(1 for r in results if r['hallucination_score'] > 0.5)
            
            print(f"\n🧠 Hallucination Analysis:")
            print(f"✅ Low Hallucination (<20%):  {low_halluc} answers")
            print(f"⚠️  Medium Hallucination (20-50%): {med_halluc} answers")
            print(f"🔴 High Hallucination (>50%): {high_halluc} answers")
            
            # Save detailed results
            results_df = pd.DataFrame(results)
            results_df.to_excel("comprehensive_evaluation_results.xlsx", index=False)
            print(f"\n💾 Detailed results saved to: 'comprehensive_evaluation_results.xlsx'")
            
            # Overall performance assessment with specific recommendations
            print(f"\n🔍 DETAILED ASSESSMENT:")
            if avg_overall > 80:
                print("🎉 Excellent! Your chatbot is performing very well across all metrics.")
            elif avg_overall > 60:
                print("👍 Good performance overall. Here are specific recommendations:")
                if avg_accuracy < 70:
                    print("   💡 Improve factual accuracy - update knowledge base or training")
                if avg_relevance < 70:
                    print("   💡 Enhance question understanding - improve retrieval system")
                if avg_completeness < 70:
                    print("   💡 Provide more detailed answers - expand response generation")
                if avg_hallucination > 30:
                    print("   💡 Reduce hallucination - restrict to knowledge base only")
            elif avg_overall > 40:
                print("⚠️  Fair performance. Major improvements needed:")
                print("   💡 Review and expand training data")
                print("   💡 Improve retrieval and response generation systems")
                if avg_hallucination > 50:
                    print("   🚨 High hallucination rate - implement strict fact checking")
            else:
                print("🔴 Poor performance. Chatbot needs complete overhaul:")
                print("   💡 Rebuild knowledge base with more comprehensive data")
                print("   💡 Implement better retrieval and generation pipeline")
                print("   💡 Add hallucination prevention mechanisms")
            
            return results
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return []

# Simple usage
def run_comprehensive_evaluation():
    """Run the comprehensive evaluation with all 4 metrics"""
    print("🌱 Starting Comprehensive Chatbot Evaluation")
    print("📊 Metrics: Accuracy + Relevance + Completeness + Hallucination")
    print("="*60)
    
    evaluator = ComprehensiveEvaluator()
    results = evaluator.evaluate_answers(
        chatbot_file="chatbot_result.xlsx",
        actual_file="actual_answers.xlsx"
    )
    
    return results

# Run this to start comprehensive evaluation
if __name__ == "__main__":
    run_comprehensive_evaluation()