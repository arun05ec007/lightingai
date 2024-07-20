from fastapi import APIRouter, status, Depends
from API.Index_Data import Index_data
from API.Chatbot_Api import Chat_bot
from API.Index_Creation import index_creation

router=APIRouter()


router.add_api_route('/Upload_File',Index_data.Extract_text,methods=["POST"])
router.add_api_route('/Chatbot',Chat_bot.gpt_query,methods=["POST"])
router.add_api_route('/Create_index',index_creation.Create_Index,methods=["POST"])
router.add_api_route('/delete_index',index_creation.Delete_Index,methods=["POST"])