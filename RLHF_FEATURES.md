# RLHF (Reinforcement Learning from Human Feedback) Features

## Overview

HumanSQL now includes an advanced RLHF system that learns from user feedback to improve SQL query generation over time. The system collects user feedback through like/dislike buttons and uses this data to generate better queries for similar questions in the future.

## New Features

### 1. **Feedback Collection System**
- **Like/Dislike Buttons**: After each SQL query is generated, users can provide feedback using thumbs up/down buttons
- **Feedback Storage**: All feedback is stored with timestamps and linked to specific queries
- **Pattern Recognition**: The system extracts patterns from questions that receive positive/negative feedback

### 2. **Improved UI Layout**
- **Natural Answer Display**: Conversational responses now appear above the query input in a dedicated green section
- **Feedback Interface**: Clean, intuitive feedback buttons below each generated SQL query
- **Better Organization**: Clear separation between natural answers and technical SQL output

### 3. **Learning Algorithm**
- **Pattern Matching**: Extracts keywords from user questions to identify similar query patterns
- **Scoring System**: Maintains feedback scores for different question patterns
- **Query Improvement**: Uses learned patterns to generate better SQL queries for similar future questions

### 4. **Database Schema Updates**
- **Enhanced Query History**: Now includes feedback columns and timestamps
- **Feedback Patterns Table**: Stores learned patterns with success scores and usage counts
- **Automatic Migration**: Tables are created/updated automatically on startup

## How It Works

### Feedback Collection
1. User asks a question in natural language
2. System generates SQL query and shows results
3. User clicks like 👍 or dislike 👎 button
4. Feedback is stored and processed for learning

### Learning Process
1. **Pattern Extraction**: Keywords are extracted from the user's question
2. **Feedback Processing**: Positive feedback increases pattern scores, negative feedback decreases them
3. **Pattern Storage**: Successful query patterns are stored with their effectiveness scores
4. **Future Improvement**: When similar questions are asked, the system references learned patterns

### Query Enhancement
1. **Pattern Matching**: New questions are matched against learned patterns
2. **Context Integration**: Successful patterns are included in the AI prompt
3. **Improved Generation**: The AI uses learned patterns to generate better SQL queries

## Technical Implementation

### Backend Changes
- **New Endpoints**:
  - `POST /feedback` - Submit user feedback
  - Enhanced `/query` - Returns query ID for feedback
  - Enhanced `/query_history` - Includes feedback data

- **Database Tables**:
  ```sql
  -- Enhanced query_history table
  ALTER TABLE query_history ADD COLUMN feedback INTEGER DEFAULT NULL;
  ALTER TABLE query_history ADD COLUMN feedback_timestamp TIMESTAMP DEFAULT NULL;
  
  -- New feedback_patterns table
  CREATE TABLE feedback_patterns (
      id SERIAL PRIMARY KEY,
      question_pattern TEXT UNIQUE,
      successful_query TEXT,
      feedback_score FLOAT DEFAULT 0.0,
      usage_count INTEGER DEFAULT 1,
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  ```

### Frontend Changes
- **New UI Components**:
  - Natural answer display section
  - Feedback buttons with icons
  - Improved layout organization

- **New State Management**:
  - `currentQueryId` - Tracks active query for feedback
  - `naturalAnswer` - Stores conversational response
  - `showNaturalAnswer` - Controls answer display

## Usage Examples

### Basic Feedback Flow
1. **Ask Question**: "Show me all my skills"
2. **Review Results**: Check the generated SQL and results
3. **Provide Feedback**: Click 👍 if the query is good, 👎 if it needs improvement
4. **System Learning**: Your feedback helps improve future similar queries

### Improved Queries Over Time
- **First Time**: User asks "What are my skills?" → Basic query generated
- **After Feedback**: User provides positive feedback on a good skills query
- **Future Queries**: Similar questions like "List my abilities" benefit from learned patterns

## Benefits

### For Users
- **Better Accuracy**: Queries improve over time based on your feedback
- **Personalized Experience**: System learns your preferences and query patterns
- **Clear Interface**: Natural answers are prominently displayed above technical details

### For System
- **Continuous Improvement**: Automatic learning without manual intervention
- **Pattern Recognition**: Identifies successful query structures for different question types
- **Quality Assurance**: User feedback helps identify and fix poor query generation

## Testing the RLHF System

Run the included test script to verify functionality:

```bash
python test_rlhf.py
```

### Manual Testing Steps
1. **Upload Data**: Upload a CSV, Excel, PDF, or text file
2. **Ask Questions**: Submit various natural language questions
3. **Provide Feedback**: Use the like/dislike buttons on generated queries
4. **Test Learning**: Ask similar questions to see if responses improve
5. **Check History**: View query history to see feedback records

## Configuration

### Environment Variables
No additional configuration required - RLHF features use the existing database connection.

### Database Requirements
- PostgreSQL database (same as existing requirements)
- Automatic table creation on first run
- No manual schema changes needed

## Monitoring and Analytics

### Feedback Metrics
- Track feedback ratios (likes vs dislikes)
- Monitor query improvement over time
- Identify common question patterns

### Query Performance
- Measure query accuracy improvements
- Track user satisfaction trends
- Analyze learning effectiveness

## Future Enhancements

### Planned Features
- **Advanced Pattern Matching**: More sophisticated similarity detection
- **User-Specific Learning**: Personalized patterns per user
- **Feedback Analytics Dashboard**: Visual feedback and improvement metrics
- **Export Learning Data**: Ability to export/import learned patterns

### Potential Improvements
- **Multi-language Support**: RLHF for different languages
- **Query Complexity Analysis**: Learn from query complexity preferences
- **Contextual Learning**: Consider table structure in pattern matching

## Troubleshooting

### Common Issues
1. **Feedback Not Saving**: Check database connection and table permissions
2. **No Query ID**: Ensure query was successfully stored in history
3. **Patterns Not Learning**: Verify feedback_patterns table exists and is writable

### Debug Steps
1. Check application logs for RLHF-related errors
2. Verify database tables exist: `query_history`, `feedback_patterns`
3. Test feedback endpoint directly with curl/Postman
4. Run the test script to verify functionality

## API Documentation

### Submit Feedback
```http
POST /feedback
Content-Type: application/json

{
  "query_id": 123,
  "feedback": 1  // 1 for like, -1 for dislike
}
```

### Response
```json
{
  "message": "Feedback recorded successfully"
}
```

---

The RLHF system represents a significant advancement in HumanSQL's ability to provide personalized, improving query generation based on real user feedback and preferences.